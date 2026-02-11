from typing import Callable, List, Tuple

import math
import random
import struct
from collections import deque

import numpy as np

from max import functional as F
from max import engine
from max.driver import CPU
from max.tensor import Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

# Discount factor for TD target
GAMMA = 0.99

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # This provides a random batch of memories to break correlation
        return random.sample(self.buffer, batch_size)

def _check_winner_static(b: list) -> int:
    """Check winner of board b. Returns 1, -1, or 0 (no winner)."""
    for i in range(3):
        if b[i*3] == b[i*3+1] == b[i*3+2] != 0:
            return b[i*3]
        if b[i] == b[i+3] == b[i+6] != 0:
            return b[i]
    if b[0] == b[4] == b[8] != 0:
        return b[0]
    if b[2] == b[4] == b[6] != 0:
        return b[2]
    return 0


def minimax_best_move(board: list, player: int) -> int:
    """Return the best move for `player` using full minimax search.
    `player` is the current player to move (1 or -1)."""

    def _minimax(b: list, is_maximizing: bool) -> int:
        """Returns score from `player`'s perspective: +1 win, -1 loss, 0 draw."""
        w = _check_winner_static(b)
        if w == player:
            return 1
        elif w == -player:
            return -1
        moves = [i for i in range(9) if b[i] == 0]
        if not moves:
            return 0
        # Determine whose turn it is
        x_count = sum(1 for c in b if c == 1)
        o_count = sum(1 for c in b if c == -1)
        current = 1 if x_count == o_count else -1

        if is_maximizing:
            best = -2
            for m in moves:
                nb = b[:]
                nb[m] = current
                score = _minimax(nb, current != player)
                if score > best:
                    best = score
            return best
        else:
            best = 2
            for m in moves:
                nb = b[:]
                nb[m] = current
                score = _minimax(nb, current != player)
                if score < best:
                    best = score
            return best

    moves = [i for i in range(9) if board[i] == 0]
    best_score = -2
    best_move = moves[0]
    for m in moves:
        nb = board[:]
        nb[m] = player
        # After player moves, it's opponent's turn => not maximizing
        score = _minimax(nb, False)
        if score > best_score:
            best_score = score
            best_move = m
    return best_move


class DataCollector:
    def __init__(self, buffer: ReplayBuffer, model, epsilon: float,
                 forced_opening_ratio: float = 0.0,
                 minimax_opponent_ratio: float = 0.0):
        self.buffer = buffer
        self.model = model
        self.epsilon = epsilon
        # Fraction of games that start with a forced random opening move
        self.forced_opening_ratio = forced_opening_ratio
        # Fraction of games played against minimax opponent
        self.minimax_opponent_ratio = minimax_opponent_ratio

    def collect_data(self, num_episodes: int):
        env = TicTacToeEnv()
        for episode in range(num_episodes):
            env.reset()
            done = False

            use_minimax_opp = random.random() < self.minimax_opponent_ratio
            # Randomly assign minimax to play as X or O
            minimax_player = random.choice([1, -1]) if use_minimax_opp else None

            # Force diverse openings: play 1-2 random moves at start
            if random.random() < self.forced_opening_ratio:
                # Force 1 or 2 random opening moves to create diverse positions
                num_forced = random.choice([1, 2])
                for _ in range(num_forced):
                    if done:
                        break
                    player = env.current_player
                    state_norm = _normalize_board(env.board, player)
                    action = random.choice(env.legal_moves())
                    next_state_raw, reward, done = env.step(action)
                    next_state_norm = _normalize_board(next_state_raw, env.current_player)
                    self.buffer.push(state_norm, action, reward, next_state_norm, done)

            while not done:
                player = env.current_player
                state_norm = _normalize_board(env.board, player)

                # Use minimax for the designated player if applicable
                if minimax_player is not None and player == minimax_player:
                    action = minimax_best_move(env.board, player)
                else:
                    action = self.select_action(env)

                next_state_raw, reward, done = env.step(action)
                # Normalize next_state from the NEXT player's perspective
                next_state_norm = _normalize_board(next_state_raw, env.current_player)
                self.buffer.push(state_norm, action, reward, next_state_norm, done)

    def select_action(self, env: "TicTacToeEnv") -> int:
        if random.random() < self.epsilon:
            return random.choice(env.legal_moves())
        state_norm = _normalize_board(env.board, env.current_player)
        state_np = _board_to_numpy(state_norm)
        prediction = self.model.execute(state_np)  # numpy (9,)
        env.mask_illegal_moves(prediction)
        return int(np.argmax(prediction))

class TicTacToeEnv:
    def __init__(self):
        self.board = [0] * 9
        self.current_player = 1
        self.winner = None
        self.done = False
    
    def reset(self):
        self.board = [0] * 9
        self.current_player = 1
        self.winner = None
        self.done = False
    
    def check_winner(self) -> int:
        for i in range(3):
            if self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2] != 0:
                return self.board[i * 3]
        for i in range(3):
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != 0:
                return self.board[i]
        if self.board[0] == self.board[4] == self.board[8] != 0:
            return self.board[0]
        if self.board[2] == self.board[4] == self.board[6] != 0:
            return self.board[2]
        return 0
    
    def is_full(self) -> bool:
        return all(self.board)

    # Returns: next_state (list), reward, done
    def step(self, action: int) -> Tuple[list, float, bool]:
        row = action // 3
        col = action % 3
        assert self.board[row * 3 + col] == 0
        self.board[row * 3 + col] = self.current_player
        self.winner = self.check_winner()
        self.done = self.winner != 0 or self.is_full()
        if self.winner == self.current_player:
            reward = 1.0
        elif self.done:  # draw
            reward = 0.0
        else:
            reward = 0.0
        self.current_player = self.current_player * -1
        return self.board.copy(), reward, self.done

    def mask_illegal_moves(self, prediction: np.ndarray) -> None:
        """Mutate prediction in place: set illegal (occupied) positions to -inf."""
        for i in range(9):
            if self.board[i] != 0:
                prediction[i] = -np.inf
    
    def legal_moves(self) -> List[int]:
        return [i for i in range(9) if self.board[i] == 0]

def _numpy_to_tensor_for_execute(arr: np.ndarray, dtype: np.dtype) -> Tensor:
    """Convert numpy array to Max Tensor for feeding into Model.execute (graph boundary)."""
    arr = np.ascontiguousarray(arr.astype(dtype))
    return Tensor.from_dlpack(arr)


def _tensor_to_numpy(t: Tensor) -> np.ndarray:
    """Convert Max Tensor output from Model.execute back to numpy."""
    if hasattr(t, "to_numpy"):
        return t.to_numpy()
    return np.from_dlpack(t)


class TTTModel:
    """TTT Q-network with biases. Weights stored as numpy; only converted to Tensor at execute/backward boundary.

    weights is a list of 6 numpy arrays: [w1, w2, w3, b1, b2, b3].
    """

    def __init__(self, weights: List[np.ndarray]):
        # 1. Build the graph (Max Tensor types only).
        input_type = TensorType(
            dtype=DType.int32, shape=(9,), device=DeviceRef.CPU()
        )
        w1t = TensorType(
            dtype=DType.float32, shape=(9, 128), device=DeviceRef.CPU()
        )
        w2t = TensorType(
            dtype=DType.float32, shape=(128, 64), device=DeviceRef.CPU()
        )
        w3t = TensorType(
            dtype=DType.float32, shape=(64, 9), device=DeviceRef.CPU()
        )
        b1t = TensorType(
            dtype=DType.float32, shape=(128,), device=DeviceRef.CPU()
        )
        b2t = TensorType(
            dtype=DType.float32, shape=(64,), device=DeviceRef.CPU()
        )
        b3t = TensorType(
            dtype=DType.float32, shape=(9,), device=DeviceRef.CPU()
        )
        with Graph(
            "ttt_model",
            input_types=(input_type, w1t, w2t, w3t, b1t, b2t, b3t),
        ) as graph:
            input = graph.inputs[0]
            w1 = graph.inputs[1]
            w2 = graph.inputs[2]
            w3 = graph.inputs[3]
            b1 = graph.inputs[4]
            b2 = graph.inputs[5]
            b3 = graph.inputs[6]
            input_f = ops.cast(input, DType.float32)
            step1 = F.relu(input_f @ w1 + b1)
            step2 = F.relu(step1 @ w2 + b2)
            step3 = step2 @ w3 + b3
            graph.output(step3)

        self.graph = graph
        self.weights = [np.asarray(w).astype(np.float32) for w in weights]

        # 2. Create an inference session.
        self.forward_session = engine.InferenceSession(devices=[CPU()])
        self.forward_model = self.forward_session.load(self.graph)

        # Build backward graph.
        d_pred_type = TensorType(
            dtype=DType.float32, shape=(9,), device=DeviceRef.CPU()
        )
        self.backward_graph = _build_ttt_backward_graph(
            input_type, w1t, w2t, w3t, b1t, b2t, b3t, d_pred_type
        )
        self.backward_session = engine.InferenceSession(devices=[CPU()])
        self.backward_model = self.backward_session.load(self.backward_graph)

    def execute(self, state: list | np.ndarray) -> np.ndarray:
        """Forward pass. state: board as list or (9,) int32 array. Returns (9,) float32 numpy."""
        state_np = _board_to_numpy(state)
        state_t = _numpy_to_tensor_for_execute(state_np, np.int32)
        w1_t = _numpy_to_tensor_for_execute(self.weights[0], np.float32)
        w2_t = _numpy_to_tensor_for_execute(self.weights[1], np.float32)
        w3_t = _numpy_to_tensor_for_execute(self.weights[2], np.float32)
        b1_t = _numpy_to_tensor_for_execute(self.weights[3], np.float32)
        b2_t = _numpy_to_tensor_for_execute(self.weights[4], np.float32)
        b3_t = _numpy_to_tensor_for_execute(self.weights[5], np.float32)
        out_t = self.forward_model.execute(state_t, w1_t, w2_t, w3_t, b1_t, b2_t, b3_t)[0]
        return _tensor_to_numpy(out_t)

    def backward(self, state: list | np.ndarray, d_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass. state and d_prediction as numpy. Returns (d_w1, d_w2, d_w3, d_b1, d_b2, d_b3) numpy."""
        state_np = _board_to_numpy(state)
        d_pred_np = np.ascontiguousarray(d_prediction.astype(np.float32))
        state_t = _numpy_to_tensor_for_execute(state_np, np.int32)
        d_pred_t = _numpy_to_tensor_for_execute(d_pred_np, np.float32)
        w1_t = _numpy_to_tensor_for_execute(self.weights[0], np.float32)
        w2_t = _numpy_to_tensor_for_execute(self.weights[1], np.float32)
        w3_t = _numpy_to_tensor_for_execute(self.weights[2], np.float32)
        b1_t = _numpy_to_tensor_for_execute(self.weights[3], np.float32)
        b2_t = _numpy_to_tensor_for_execute(self.weights[4], np.float32)
        b3_t = _numpy_to_tensor_for_execute(self.weights[5], np.float32)
        outputs = self.backward_model.execute(state_t, w1_t, w2_t, w3_t, b1_t, b2_t, b3_t, d_pred_t)
        return (
            _tensor_to_numpy(outputs[0]), _tensor_to_numpy(outputs[1]), _tensor_to_numpy(outputs[2]),
            _tensor_to_numpy(outputs[3]), _tensor_to_numpy(outputs[4]), _tensor_to_numpy(outputs[5]),
        )


def _build_ttt_backward_graph(
    input_type: TensorType,
    w1t: TensorType,
    w2t: TensorType,
    w3t: TensorType,
    b1t: TensorType,
    b2t: TensorType,
    b3t: TensorType,
    d_pred_type: TensorType,
) -> Graph:
    """Build the backward graph for the TTT model using the same forward-style APIs.

    The backward graph takes (input, w1, w2, w3, b1, b2, b3, d_prediction) and outputs
    (d_w1, d_w2, d_w3, d_b1, d_b2, d_b3). It recomputes forward activations inside
    the graph and applies the chain rule using only max.graph.ops (matmul, relu, mul,
    greater, cast, transpose, unsqueeze, squeeze).
    """
    with Graph(
        "ttt_backward",
        input_types=(input_type, w1t, w2t, w3t, b1t, b2t, b3t, d_pred_type),
    ) as graph:
        input_val = graph.inputs[0]
        w1 = graph.inputs[1]
        w2 = graph.inputs[2]
        w3 = graph.inputs[3]
        b1 = graph.inputs[4]
        b2 = graph.inputs[5]
        b3 = graph.inputs[6]
        d_prediction = graph.inputs[7]

        # Cast input to float32 for gradient computation (forward uses int32)
        input_f = ops.cast(input_val, DType.float32)

        # Recompute forward activations (same ops as forward, now with biases)
        pre_step1 = input_f @ w1 + b1
        step1 = F.relu(pre_step1)
        pre_step2 = step1 @ w2 + b2
        step2 = F.relu(pre_step2)

        # d_loss / d_step3  (output layer: step3 = step2 @ w3 + b3)
        d_step3 = d_prediction

        # d_b3 = d_step3  (9,)
        d_b3 = d_step3

        # d_w3 = step2.T @ d_step3  (step2: (64,), d_step3: (9,) -> (64, 9))
        step2_2d = ops.unsqueeze(step2, 1)      # (64, 1)
        d_step3_2d = ops.unsqueeze(d_step3, 0)  # (1, 9)
        d_w3 = ops.matmul(step2_2d, d_step3_2d)  # (64, 9)

        # d_step2 = d_step3 @ w3.T  (d_step3: (9,), w3: (64,9) -> w3.T (9,64) -> (64,))
        w3_t = ops.transpose(w3, 0, 1)  # (9, 64)
        d_step2_2d = ops.matmul(d_step3_2d, w3_t)  # (1, 9) @ (9, 64) = (1, 64)
        d_step2 = ops.squeeze(d_step2_2d, 0)    # (64,)

        # ReLU backward for step2: d_pre_step2 = d_step2 * (pre_step2 > 0)
        zero = ops.constant(0.0, DType.float32, DeviceRef.CPU())
        relu_mask_2 = ops.cast(ops.greater(pre_step2, zero), DType.float32)
        d_pre_step2 = ops.mul(d_step2, relu_mask_2)

        # d_b2 = d_pre_step2  (64,)
        d_b2 = d_pre_step2

        # d_w2 = step1.T @ d_pre_step2  (128, 64)
        step1_2d = ops.unsqueeze(step1, 1)         # (128, 1)
        d_pre_step2_2d = ops.unsqueeze(d_pre_step2, 0)  # (1, 64)
        d_w2 = ops.matmul(step1_2d, d_pre_step2_2d)  # (128, 64)

        # d_step1 = d_pre_step2 @ w2.T  (64,) @ (64, 128) -> (128,)
        w2_t = ops.transpose(w2, 0, 1)  # (64, 128)
        d_step1_2d = ops.matmul(d_pre_step2_2d, w2_t)  # (1, 64) @ (64, 128) = (1, 128)
        d_step1 = ops.squeeze(d_step1_2d, 0)  # (128,)

        # ReLU backward for step1
        relu_mask_1 = ops.cast(ops.greater(pre_step1, zero), DType.float32)
        d_pre_step1 = ops.mul(d_step1, relu_mask_1)

        # d_b1 = d_pre_step1  (128,)
        d_b1 = d_pre_step1

        # d_w1 = input_f.T @ d_pre_step1  (9, 128)
        input_f_2d = ops.unsqueeze(input_f, 1)   # (9, 1)
        d_pre_step1_2d = ops.unsqueeze(d_pre_step1, 0)  # (1, 128)
        d_w1 = ops.matmul(input_f_2d, d_pre_step1_2d)  # (9, 128)

        graph.output(d_w1, d_w2, d_w3, d_b1, d_b2, d_b3)
    return graph


def _normalize_board(board: list, player: int) -> list:
    """Normalize board from the given player's perspective.

    The current player's pieces become +1, opponent's become -1, empty stays 0.
    This lets a single Q-network learn a unified policy for both sides.
    """
    return [cell * player for cell in board]


def _board_to_numpy(board: list | np.ndarray) -> np.ndarray:
    """Convert a board (list of 9 ints or array) to numpy int32 shape (9,)."""
    return np.asarray(board, dtype=np.int32).reshape(9)


def compute_td_targets(
    model: "TTTModel",
    next_states: list,
    rewards: list,
    dones: list,
) -> list:
    """
    For each transition, compute the TD target.

    next_states are already normalized from the NEXT player's (opponent's)
    perspective.  In a two-player zero-sum game the opponent's best Q-value
    is bad for us, so we negate it:

        target = r  - gamma * max_a Q(s'_opponent, a)   (if not done)
        target = r                                       (if done)
    """
    targets = []
    for ns, r, done in zip(next_states, rewards, dones):
        if done:
            targets.append(float(r))
            continue
        q_next = model.execute(ns)  # numpy (9,) – from opponent's view
        # Mask illegal (occupied) positions so max is over legal moves only
        for i in range(9):
            if ns[i] != 0:
                q_next[i] = -np.inf
        max_next_q = float(np.max(q_next))
        targets.append(r - GAMMA * max_next_q)
    return targets


def train_model(
    num_episodes: int = 1000,
    num_train_steps: int = 5000,
    batch_size: int = 100,
    learning_rate: float = 1e-3,
    lr_decay: float = 0.99999,
    buffer_capacity: int = 10000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    log_interval: int = 50,
    progress_callback: Callable[[int, float, float], None] | None = None,
    weights_path: str | None = None,
    episodes_per_step: int = 10,
    target_tau: float = 0.005,
    initial_weights: List[np.ndarray] | None = None,
    grad_clip: float = 1.0,
    forced_opening_ratio: float = 0.5,
    minimax_opponent_ratio: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train the TTT Q-network. Returns (w1, w2, w3, b1, b2, b3) as numpy.
    If progress_callback(step, loss, step_time_sec) is provided, it is called at log_interval.
    If weights_path is set, weights are saved at each log_interval and on Ctrl-C.
    If episodes_per_step > 0, that many episodes are collected with the current policy before
    each training step (interleaves data collection and training).
    Epsilon decays from epsilon_start to epsilon_end over training (multiplicative decay).
    Learning rate decays by lr_decay each step.
    A target network is used for stable TD targets, soft-updated each step with
    blending factor target_tau (Polyak averaging).
    If initial_weights is provided ([w1, w2, w3, b1, b2, b3]), training resumes from those
    weights instead of random initialization.
    grad_clip: max gradient norm for clipping (0 = no clipping).
    forced_opening_ratio: fraction of self-play games starting with random forced opening moves.
    minimax_opponent_ratio: fraction of games played against minimax opponent.
    """
    if initial_weights is not None:
        w1, w2, w3, b1, b2, b3 = initial_weights
    else:
        w1 = _kaiming_uniform([9, 128])
        w2 = _kaiming_uniform([128, 64])
        w3 = _kaiming_uniform([64, 9])
        b1 = np.zeros(128, dtype=np.float32)
        b2 = np.zeros(64, dtype=np.float32)
        b3 = np.zeros(9, dtype=np.float32)
    model = TTTModel([w1, w2, w3, b1, b2, b3])

    # Target network: a frozen snapshot of the weights used for stable TD targets.
    target_weights = [w.copy() for w in model.weights]

    # Adam optimizer state (per-parameter momentum and variance)
    adam_m = [np.zeros_like(w) for w in model.weights]  # first moment
    adam_v = [np.zeros_like(w) for w in model.weights]  # second moment
    adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-8

    buffer = ReplayBuffer(buffer_capacity)
    data_collector = DataCollector(buffer, model, epsilon_start,
                                   forced_opening_ratio=forced_opening_ratio,
                                   minimax_opponent_ratio=minimax_opponent_ratio)
    data_collector.collect_data(num_episodes)

    try:
        for step in range(num_train_steps):
            # Epsilon decay: explore heavily early, exploit later
            epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** step))
            data_collector.epsilon = epsilon

            if episodes_per_step > 0:
                data_collector.collect_data(episodes_per_step)

            if len(buffer.buffer) < batch_size:
                continue

            step_start = _time_ms()
            samples = buffer.sample(batch_size)
            states = [s[0] for s in samples]
            actions = [s[1] for s in samples]
            rewards = [s[2] for s in samples]
            next_states = [s[3] for s in samples]
            dones = [s[4] for s in samples]

            # Forward pass (numpy in/out; model.execute converts at boundary).
            predictions = [model.execute(st) for st in states]
            prediction = np.stack(predictions, axis=0).astype(np.float32)

            # Use frozen target weights for TD target computation (stability).
            online_weights = model.weights
            model.weights = target_weights
            td_targets = compute_td_targets(model, next_states, rewards, dones)
            model.weights = online_weights

            target = prediction.copy()
            for j in range(len(actions)):
                target[j, actions[j]] = td_targets[j]

            # Loss: mean over batch of per-action squared error
            loss_mse = float(np.sum((prediction - target) ** 2) / batch_size)

            d_pred = 2.0 * (prediction - target) / batch_size

            d_w1_acc = np.zeros((9, 128), dtype=np.float32)
            d_w2_acc = np.zeros((128, 64), dtype=np.float32)
            d_w3_acc = np.zeros((64, 9), dtype=np.float32)
            d_b1_acc = np.zeros(128, dtype=np.float32)
            d_b2_acc = np.zeros(64, dtype=np.float32)
            d_b3_acc = np.zeros(9, dtype=np.float32)
            for j in range(batch_size):
                d_w1_j, d_w2_j, d_w3_j, d_b1_j, d_b2_j, d_b3_j = model.backward(states[j], d_pred[j])
                d_w1_acc += d_w1_j
                d_w2_acc += d_w2_j
                d_w3_acc += d_w3_j
                d_b1_acc += d_b1_j
                d_b2_acc += d_b2_j
                d_b3_acc += d_b3_j

            grads = [d_w1_acc, d_w2_acc, d_w3_acc, d_b1_acc, d_b2_acc, d_b3_acc]

            # Gradient clipping by global norm
            if grad_clip > 0:
                global_norm = math.sqrt(sum(float(np.sum(g ** 2)) for g in grads))
                if global_norm > grad_clip:
                    scale = grad_clip / global_norm
                    grads = [g * scale for g in grads]

            # Learning rate with decay
            lr = learning_rate * (lr_decay ** step)

            # Adam update
            t_adam = step + 1
            weights_list = [w1, w2, w3, b1, b2, b3]
            for i in range(6):
                adam_m[i] = adam_beta1 * adam_m[i] + (1 - adam_beta1) * grads[i]
                adam_v[i] = adam_beta2 * adam_v[i] + (1 - adam_beta2) * (grads[i] ** 2)
                m_hat = adam_m[i] / (1 - adam_beta1 ** t_adam)
                v_hat = adam_v[i] / (1 - adam_beta2 ** t_adam)
                weights_list[i] = weights_list[i] - lr * m_hat / (np.sqrt(v_hat) + adam_eps)

            w1, w2, w3, b1, b2, b3 = weights_list
            model.weights = [w1, w2, w3, b1, b2, b3]

            # Soft (Polyak) target network update each step
            target_weights = [
                target_tau * w + (1.0 - target_tau) * tw
                for w, tw in zip(model.weights, target_weights)
            ]

            step_time_sec = (_time_ms() - step_start) / 1000.0
            if (step + 1) % log_interval == 0 or step == 0:
                if progress_callback:
                    progress_callback(step + 1, loss_mse, step_time_sec)
                if weights_path:
                    save_weights(w1, w2, w3, b1, b2, b3, weights_path)
    except KeyboardInterrupt:
        if weights_path:
            print("\nInterrupted. Saving current weights to", weights_path, "...")
            save_weights(w1, w2, w3, b1, b2, b3, weights_path)
            print("Saved.")
        raise
    return w1, w2, w3, b1, b2, b3


def _kaiming_uniform(shape: List[int]) -> np.ndarray:
    """Kaiming (He) uniform init for ReLU layers: U(-bound, bound), bound = sqrt(6 / fan_in)."""
    fan_in = shape[0]
    bound = math.sqrt(6.0 / float(fan_in))
    return np.random.uniform(-bound, bound, size=shape).astype(np.float32)


def _time_ms() -> float:
    """Monotonic time in milliseconds (for measuring step duration)."""
    import time
    return time.monotonic() * 1000.0


def save_weights(
    w1: np.ndarray, w2: np.ndarray, w3: np.ndarray,
    b1: np.ndarray, b2: np.ndarray, b3: np.ndarray,
    path: str,
) -> None:
    """Save weight and bias arrays to a binary file. Can be loaded with load_weights()."""
    with open(path, "wb") as f:
        for w in (w1, w2, w3):
            w = np.asarray(w).astype(np.float32)
            n, m = w.shape[0], w.shape[1]
            f.write(struct.pack("II", n, m))
            f.write(w.tobytes())
        for b in (b1, b2, b3):
            b = np.asarray(b).astype(np.float32).ravel()
            n = len(b)
            f.write(struct.pack("I", n))
            f.write(b.tobytes())


def load_weights(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load weight and bias arrays from a file saved by save_weights()."""
    weights = []
    biases = []
    with open(path, "rb") as f:
        for _ in range(3):
            n, m = struct.unpack("II", f.read(8))
            data = np.frombuffer(f.read(n * m * 4), dtype=np.float32).reshape(n, m).copy()
            weights.append(data)
        for _ in range(3):
            n, = struct.unpack("I", f.read(4))
            data = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
            biases.append(data)
    return weights[0], weights[1], weights[2], biases[0], biases[1], biases[2]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train TTT Q-network")
    parser.add_argument("weights_path", nargs="?", default="ttt_weights.bin",
                        help="Path to save/load weights (default: ttt_weights.bin)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing weights file")
    args = parser.parse_args()

    import sys
    weights_path = args.weights_path
    log_interval = 100
    num_episodes = 2000          # More initial episodes for diverse buffer fill
    num_train_steps = 40000      # More steps to learn all positions
    episodes_per_step = 10
    epsilon_decay = 0.99985      # Slower decay => more exploration over 40k steps
    batch_size = 128
    epsilon_start = 1.0
    epsilon_end = 0.08           # Higher floor for continued exploration
    learning_rate = 5e-4         # Slightly lower LR for stability with Adam
    lr_decay = 0.99999
    buffer_capacity = 50000      # Much larger buffer for diverse experiences
    target_tau = 0.005
    grad_clip = 1.0              # Gradient clipping for stability
    forced_opening_ratio = 0.5   # 50% of games start with random forced openings
    minimax_opponent_ratio = 0.3 # 30% of games against perfect minimax

    initial_weights = None
    if args.resume:
        print("Resuming from", weights_path, "...")
        w1, w2, w3, b1, b2, b3 = load_weights(weights_path)
        initial_weights = [w1, w2, w3, b1, b2, b3]
        # When resuming, start with moderate exploration to fix weak spots
        epsilon_start = 0.4

    def on_progress(step: int, loss: float, step_time_sec: float) -> None:
        print(f"  step {step:5d} / {num_train_steps}  loss = {loss:.6f}  step_time = {step_time_sec:.3f}s  eps={max(epsilon_end, epsilon_start * (epsilon_decay ** step)):.4f}")
        sys.stdout.flush()

    print("Training TTT Q-network ...")
    print(f"  Episodes: {num_episodes}, Train steps: {num_train_steps}, Batch: {batch_size}")
    print(f"  LR: {learning_rate}, Epsilon: {epsilon_start}->{epsilon_end} (decay={epsilon_decay})")
    print(f"  Buffer: {buffer_capacity}, Forced openings: {forced_opening_ratio}, Minimax opp: {minimax_opponent_ratio}")
    print(f"  Grad clip: {grad_clip}, Target tau: {target_tau}, Adam optimizer")
    sys.stdout.flush()
    w1, w2, w3, b1, b2, b3 = train_model(
        num_episodes=num_episodes,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        buffer_capacity=buffer_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        log_interval=log_interval,
        progress_callback=on_progress,
        weights_path=weights_path,
        episodes_per_step=episodes_per_step,
        target_tau=target_tau,
        initial_weights=initial_weights,
        grad_clip=grad_clip,
        forced_opening_ratio=forced_opening_ratio,
        minimax_opponent_ratio=minimax_opponent_ratio,
    )
    print("Saving weights to", weights_path)
    save_weights(w1, w2, w3, b1, b2, b3, weights_path)
    print("Done.")


if __name__ == "__main__":
    main()
