"""
Evaluate the tic-tac-toe neural network model by playing it against
a perfect minimax opponent and various strategic opponents.

Reports win/loss/draw statistics from the evaluator's perspective.
"""

import sys
import numpy as np
from functools import lru_cache
from ttt import load_weights, TTTModel, TicTacToeEnv, _board_to_numpy, _normalize_board


# ──────────────────────────────────────────────────────────────
# Perfect minimax player
# ──────────────────────────────────────────────────────────────

def minimax(board_tuple, player):
    """
    Returns (score, best_move) from the perspective of `player`.
    Score: +1 if player wins, -1 if player loses, 0 for draw.
    """
    board = list(board_tuple)
    winner = check_winner(board)
    if winner == player:
        return (1, None)
    elif winner == -player:
        return (-1, None)
    elif all(c != 0 for c in board):
        return (0, None)

    best_score = -2
    best_move = None
    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score, _ = minimax(tuple(board), -player)
            score = -score  # opponent's score negated = our score
            board[i] = 0
            if score > best_score:
                best_score = score
                best_move = i
    return (best_score, best_move)


# Cache for performance
_minimax_cache = {}

def minimax_cached(board_tuple, player):
    key = (board_tuple, player)
    if key in _minimax_cache:
        return _minimax_cache[key]
    result = minimax(board_tuple, player)
    _minimax_cache[key] = result
    return result


def check_winner(board):
    """Check if there's a winner. Returns 1, -1, or 0."""
    for i in range(3):
        if board[i*3] == board[i*3+1] == board[i*3+2] != 0:
            return board[i*3]
    for i in range(3):
        if board[i] == board[i+3] == board[i+6] != 0:
            return board[i]
    if board[0] == board[4] == board[8] != 0:
        return board[0]
    if board[2] == board[4] == board[6] != 0:
        return board[2]
    return 0


def model_select_move(model, env):
    """Select the model's best move given the current environment state."""
    state_norm = _normalize_board(env.board, env.current_player)
    state_np = _board_to_numpy(state_norm)
    prediction = model.execute(state_np)
    env.mask_illegal_moves(prediction)
    return int(np.argmax(prediction))


def play_game(model, model_plays_as, verbose=False):
    """
    Play a single game. model_plays_as is 1 (X, first) or -1 (O, second).
    The other side is played by perfect minimax.
    
    Returns: (result_for_evaluator, move_log)
      result_for_evaluator: 'win', 'loss', or 'draw'
      move_log: list of (player, move_idx, board_before) tuples
    """
    env = TicTacToeEnv()
    env.reset()
    move_log = []
    evaluator_player = -model_plays_as  # the minimax player

    while not env.done:
        board_before = env.board.copy()
        
        if env.current_player == model_plays_as:
            # Model's turn
            action = model_select_move(model, env)
            who = "MODEL"
        else:
            # Minimax's turn (evaluator)
            score, action = minimax_cached(tuple(env.board), env.current_player)
            who = "MINIMAX"
        
        move_log.append((who, action, board_before))
        
        if verbose:
            print(f"  {who} plays position {action} (board: {board_before})")
        
        env.step(action)

    if env.winner == evaluator_player:
        result = 'win'
    elif env.winner == model_plays_as:
        result = 'loss'
    else:
        result = 'draw'
    
    if verbose:
        print(f"  Result: {result} (winner={env.winner})")
        print_board(env.board)
    
    return result, move_log


def print_board(board):
    """Print a simple text representation of the board."""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    for row in range(3):
        cells = [symbols[board[row*3 + col]] for col in range(3)]
        print("  " + " ".join(cells))


def play_game_model_vs_random(model, model_plays_as, seed=None):
    """
    Play model vs random opponent for comparison.
    Returns result for the random player's perspective.
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)
    
    env = TicTacToeEnv()
    env.reset()
    random_player = -model_plays_as

    while not env.done:
        if env.current_player == model_plays_as:
            action = model_select_move(model, env)
        else:
            legal = env.legal_moves()
            action = rng.choice(legal)
        env.step(action)

    if env.winner == random_player:
        return 'win'
    elif env.winner == model_plays_as:
        return 'loss'
    else:
        return 'draw'


def enumerate_all_first_moves(model, model_plays_as):
    """
    For each possible first move by the first player, play to completion
    with minimax and report results. This tests model across all openings.
    """
    results = {}
    for first_move in range(9):
        env = TicTacToeEnv()
        env.reset()
        
        evaluator_player = -model_plays_as
        
        # First move
        if env.current_player == model_plays_as:
            # Model goes first - we force this specific opening
            env.step(first_move)
            # Now minimax responds
        else:
            # Minimax goes first with this specific opening
            env.step(first_move)
        
        # Play out the rest with model vs minimax
        while not env.done:
            if env.current_player == model_plays_as:
                action = model_select_move(model, env)
            else:
                score, action = minimax_cached(tuple(env.board), env.current_player)
            env.step(action)
        
        if env.winner == evaluator_player:
            results[first_move] = 'evaluator_wins'
        elif env.winner == model_plays_as:
            results[first_move] = 'model_wins'
        else:
            results[first_move] = 'draw'
    
    return results


def analyze_model_q_values(model):
    """Analyze model Q-values for key positions to understand its strategic awareness."""
    print("\n" + "="*70)
    print("STRATEGIC ANALYSIS: Q-value inspection for key positions")
    print("="*70)
    
    # Empty board - what does the model think?
    print("\n--- Empty board (model as X, first mover) ---")
    state = [0]*9
    state_norm = _normalize_board(state, 1)
    state_np = _board_to_numpy(state_norm)
    pred = model.execute(state_np)
    print("  Q-values:")
    for row in range(3):
        vals = [f"{pred[row*3+col]:+.3f}" for col in range(3)]
        print("  " + "  ".join(vals))
    preferred = int(np.argmax(pred))
    print(f"  Preferred opening: position {preferred} (corner={preferred in [0,2,6,8]}, center={preferred==4})")
    
    # Classic fork setup: X in corner (0), O in center (4) - what does X play?
    print("\n--- X=corner(0), O=center(4), X's turn ---")
    state = [1,0,0, 0,-1,0, 0,0,0]
    state_norm = _normalize_board(state, 1)
    state_np = _board_to_numpy(state_norm)
    pred = model.execute(state_np)
    for i in range(9):
        if state[i] != 0:
            pred[i] = float('-inf')
    print("  Q-values for legal moves:")
    for row in range(3):
        vals = []
        for col in range(3):
            idx = row*3+col
            if state[idx] != 0:
                vals.append("  ---  ")
            else:
                vals.append(f"{pred[idx]:+.3f}")
        print("  " + "  ".join(vals))
    preferred = int(np.argmax(pred))
    print(f"  Preferred: position {preferred} (opposite corner={preferred in [2,6,8]})")
    
    # Must-block scenario: O must block X's winning threat
    print("\n--- X has [0,1], O must block position 2 (O's turn) ---")
    state = [1,1,0, 0,-1,0, 0,0,0]
    state_norm = _normalize_board(state, -1)  # O's perspective
    state_np = _board_to_numpy(state_norm)
    pred = model.execute(state_np)
    for i in range(9):
        if state[i] != 0:
            pred[i] = float('-inf')
    preferred = int(np.argmax(pred))
    print(f"  Model (as O) prefers position {preferred} (correct=2, blocks={preferred==2})")
    
    # Must-win scenario: X has [0,1] and position 2 is open
    print("\n--- X has [0,1], position 2 wins (X's turn) ---")
    state = [1,1,0, -1,0,0, 0,-1,0]
    state_norm = _normalize_board(state, 1)
    state_np = _board_to_numpy(state_norm)
    pred = model.execute(state_np)
    for i in range(9):
        if state[i] != 0:
            pred[i] = float('-inf')
    preferred = int(np.argmax(pred))
    print(f"  Model (as X) prefers position {preferred} (correct=2, wins={preferred==2})")
    
    # Fork detection: X at 0 and 8, O at 4 - X can fork with 2 or 6
    print("\n--- X at corners 0,8; O at center 4 - X can fork ---")
    state = [1,0,0, 0,-1,0, 0,0,1]
    state_norm = _normalize_board(state, 1)
    state_np = _board_to_numpy(state_norm)
    pred = model.execute(state_np)
    for i in range(9):
        if state[i] != 0:
            pred[i] = float('-inf')
    preferred = int(np.argmax(pred))
    is_fork = preferred in [2, 6]
    print(f"  Model (as X) prefers position {preferred} (fork positions: 2,6; creates_fork={is_fork})")


def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "ttt_weights.bin"
    print(f"Loading model from {weights_path}...")
    w1, w2, w3, b1, b2, b3 = load_weights(weights_path)
    model = TTTModel([w1, w2, w3, b1, b2, b3])
    print("Model loaded.\n")

    # ── Test 1: Model as X (first player) vs Perfect Minimax ──
    print("="*70)
    print("TEST 1: Model plays X (first) vs Perfect Minimax")
    print("="*70)
    
    # Play a single verbose game
    print("\n--- Verbose game (Model=X first) ---")
    result, log = play_game(model, model_plays_as=1, verbose=True)
    print(f"Result: Evaluator {result}")
    
    # Enumerate all openings when model is X
    print("\n--- All possible model openings (Model=X) ---")
    opening_results = enumerate_all_first_moves(model, model_plays_as=1)
    for move, res in sorted(opening_results.items()):
        labels = {0:'TL', 1:'TC', 2:'TR', 3:'ML', 4:'CC', 5:'MR', 6:'BL', 7:'BC', 8:'BR'}
        print(f"  Model opens at {move} ({labels[move]}): {res}")
    
    model_x_wins = sum(1 for r in opening_results.values() if r == 'model_wins')
    eval_wins = sum(1 for r in opening_results.values() if r == 'evaluator_wins')
    draws = sum(1 for r in opening_results.values() if r == 'draw')
    print(f"\n  Summary (Model=X openings): Model wins={model_x_wins}, Evaluator wins={eval_wins}, Draws={draws}")

    # ── Test 2: Model as O (second player) vs Perfect Minimax ──
    print("\n" + "="*70)
    print("TEST 2: Model plays O (second) vs Perfect Minimax")
    print("="*70)
    
    print("\n--- Verbose game (Model=O second) ---")
    result, log = play_game(model, model_plays_as=-1, verbose=True)
    print(f"Result: Evaluator {result}")
    
    # Enumerate all openings when minimax is X (model is O)
    print("\n--- All possible minimax openings (Model=O) ---")
    opening_results_o = enumerate_all_first_moves(model, model_plays_as=-1)
    for move, res in sorted(opening_results_o.items()):
        labels = {0:'TL', 1:'TC', 2:'TR', 3:'ML', 4:'CC', 5:'MR', 6:'BL', 7:'BC', 8:'BR'}
        print(f"  Minimax opens at {move} ({labels[move]}): {res}")
    
    model_o_wins = sum(1 for r in opening_results_o.values() if r == 'model_wins')
    eval_wins_o = sum(1 for r in opening_results_o.values() if r == 'evaluator_wins')
    draws_o = sum(1 for r in opening_results_o.values() if r == 'draw')
    print(f"\n  Summary (Model=O responses): Model wins={model_o_wins}, Evaluator wins={eval_wins_o}, Draws={draws_o}")

    # ── Test 3: Model vs Random (100 games each side) ──
    print("\n" + "="*70)
    print("TEST 3: Model vs Random opponent (100 games each side)")
    print("="*70)
    
    import random as rng
    
    # Model as X vs Random
    results_x = {'win': 0, 'loss': 0, 'draw': 0}
    for i in range(100):
        r = play_game_model_vs_random(model, model_plays_as=1, seed=i*17+3)
        results_x[r] += 1
    print(f"\n  Model=X vs Random: Random wins={results_x['win']}, Model wins={results_x['loss']}, Draws={results_x['draw']}")
    
    # Model as O vs Random
    results_o = {'win': 0, 'loss': 0, 'draw': 0}
    for i in range(100):
        r = play_game_model_vs_random(model, model_plays_as=-1, seed=i*23+7)
        results_o[r] += 1
    print(f"  Model=O vs Random: Random wins={results_o['win']}, Model wins={results_o['loss']}, Draws={results_o['draw']}")

    # ── Test 4: Q-value analysis ──
    analyze_model_q_values(model)

    # ── Final Summary ──
    print("\n" + "="*70)
    print("FINAL EVALUATION SUMMARY")
    print("="*70)
    
    total_minimax_games = 18  # 9 openings * 2 sides
    total_eval_wins = eval_wins + eval_wins_o
    total_model_wins_vs_minimax = model_x_wins + model_o_wins
    total_draws_minimax = draws + draws_o
    
    print(f"\n  vs Perfect Minimax ({total_minimax_games} games across all openings):")
    print(f"    Evaluator (minimax) wins: {total_eval_wins}")
    print(f"    Model wins:               {total_model_wins_vs_minimax}")
    print(f"    Draws:                     {total_draws_minimax}")
    
    total_random = 200
    total_model_wins_random = results_x['loss'] + results_o['loss']
    total_random_wins = results_x['win'] + results_o['win']
    total_draws_random = results_x['draw'] + results_o['draw']
    
    print(f"\n  vs Random ({total_random} games):")
    print(f"    Random wins:  {total_random_wins}")
    print(f"    Model wins:   {total_model_wins_random}")
    print(f"    Draws:        {total_draws_random}")
    
    # Compute strength rating
    minimax_draw_rate = total_draws_minimax / total_minimax_games
    minimax_loss_rate = total_eval_wins / total_minimax_games  # model loses
    random_win_rate = total_model_wins_random / total_random
    
    print(f"\n  Key metrics:")
    print(f"    Draw rate vs minimax:     {minimax_draw_rate:.1%}")
    print(f"    Loss rate vs minimax:     {minimax_loss_rate:.1%}")
    print(f"    Win rate vs random:       {random_win_rate:.1%}")
    
    # Strength rating
    if minimax_draw_rate == 1.0:
        strength = 10
        level = "PERFECT (Expert)"
    elif minimax_draw_rate >= 0.8:
        strength = 8
        level = "Strong (Advanced)"
    elif minimax_draw_rate >= 0.5:
        strength = 6
        level = "Decent (Intermediate)"
    elif random_win_rate >= 0.8:
        strength = 5
        level = "Competent (beats random consistently)"
    elif random_win_rate >= 0.5:
        strength = 4
        level = "Basic (beats random sometimes)"
    elif minimax_loss_rate < 1.0:
        strength = 3
        level = "Weak (occasional draws vs minimax)"
    else:
        strength = 2
        level = "Very Weak (loses to everything)"
    
    print(f"\n  STRENGTH RATING: {strength}/10 - {level}")
    print()


if __name__ == "__main__":
    main()
