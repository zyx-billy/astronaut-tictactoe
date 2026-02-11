"""
Detailed analysis of the model's losses against minimax.
Shows move-by-move where the model goes wrong.
"""

import sys
import numpy as np
from ttt import load_weights, TTTModel, TicTacToeEnv, _board_to_numpy, _normalize_board


_minimax_cache = {}

def check_winner(board):
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

def minimax(board_tuple, player):
    key = (board_tuple, player)
    if key in _minimax_cache:
        return _minimax_cache[key]
    board = list(board_tuple)
    winner = check_winner(board)
    if winner == player:
        result = (1, None)
    elif winner == -player:
        result = (-1, None)
    elif all(c != 0 for c in board):
        result = (0, None)
    else:
        best_score = -2
        best_move = None
        for i in range(9):
            if board[i] == 0:
                board[i] = player
                score, _ = minimax(tuple(board), -player)
                score = -score
                board[i] = 0
                if score > best_score:
                    best_score = score
                    best_move = i
        result = (best_score, best_move)
    _minimax_cache[key] = result
    return result


def all_minimax_moves(board_tuple, player):
    """Return all moves that achieve the optimal minimax score."""
    best_score, _ = minimax(board_tuple, player)
    board = list(board_tuple)
    optimal_moves = []
    for i in range(9):
        if board[i] == 0:
            board[i] = player
            score, _ = minimax(tuple(board), -player)
            score = -score
            board[i] = 0
            if score == best_score:
                optimal_moves.append(i)
    return best_score, optimal_moves


def model_select_move(model, board, player):
    state_norm = _normalize_board(board, player)
    state_np = _board_to_numpy(state_norm)
    prediction = model.execute(state_np)
    for i in range(9):
        if board[i] != 0:
            prediction[i] = -np.inf
    return int(np.argmax(prediction)), prediction


def print_board(board, highlight=None):
    symbols = {0: '.', 1: 'X', -1: 'O'}
    for row in range(3):
        cells = []
        for col in range(3):
            idx = row*3+col
            s = symbols[board[idx]]
            if highlight is not None and idx == highlight:
                cells.append(f"[{s}]")
            else:
                cells.append(f" {s} ")
        print("    " + "|".join(cells))
        if row < 2:
            print("    " + "-"*11)


def analyze_loss(model, model_plays_as, first_move):
    """Play out a game with the given first move and analyze each model move."""
    labels = {0:'TL', 1:'TC', 2:'TR', 3:'ML', 4:'CC', 5:'MR', 6:'BL', 7:'BC', 8:'BR'}
    
    env = TicTacToeEnv()
    env.reset()
    
    # Make first move
    env.step(first_move)
    move_num = 1
    
    print(f"\n  Move 1: {'MODEL' if model_plays_as == 1 else 'MINIMAX'} plays {first_move} ({labels[first_move]})")
    print_board(env.board)
    
    while not env.done:
        move_num += 1
        board_before = env.board.copy()
        
        if env.current_player == model_plays_as:
            action, qvals = model_select_move(model, env.board, env.current_player)
            # Check if model's move is optimal
            opt_score, opt_moves = all_minimax_moves(tuple(env.board), env.current_player)
            is_optimal = action in opt_moves
            
            # Show Q-values for legal moves
            legal_qvals = {}
            for i in range(9):
                if board_before[i] == 0:
                    legal_qvals[i] = qvals[i]
            
            status = "OPTIMAL" if is_optimal else "MISTAKE"
            print(f"\n  Move {move_num}: MODEL plays {action} ({labels[action]}) [{status}]")
            if not is_optimal:
                print(f"    *** Optimal moves were: {[f'{m}({labels[m]})' for m in opt_moves]} (minimax score={opt_score})")
                # What score does model's move lead to?
                test_board = list(board_before)
                test_board[action] = env.current_player
                actual_score, _ = minimax(tuple(test_board), -env.current_player)
                actual_score = -actual_score  # from current player's view
                print(f"    *** Model's move leads to score={actual_score} (optimal={opt_score})")
            print(f"    Q-values: {', '.join(f'{k}({labels[k]})={v:.3f}' for k,v in sorted(legal_qvals.items()))}")
        else:
            score, action = minimax(tuple(env.board), env.current_player)
            print(f"\n  Move {move_num}: MINIMAX plays {action} ({labels[action]})")
        
        env.step(action)
        print_board(env.board)
    
    if env.winner == model_plays_as:
        print(f"\n  >>> MODEL WINS")
    elif env.winner == -model_plays_as:
        print(f"\n  >>> MODEL LOSES")
    else:
        print(f"\n  >>> DRAW")


def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "ttt_weights.bin"
    print(f"Loading model from {weights_path}...")
    w1, w2, w3, b1, b2, b3 = load_weights(weights_path)
    model = TTTModel([w1, w2, w3, b1, b2, b3])
    print("Model loaded.\n")
    
    # Analyze Model=X losses (model opens, then minimax responds)
    print("="*70)
    print("DETAILED ANALYSIS: Model=X losses")
    print("="*70)
    
    losses_x = []
    for first_move in range(9):
        # Simulate the game
        env = TicTacToeEnv()
        env.reset()
        env.step(first_move)  # model's opening
        while not env.done:
            if env.current_player == 1:  # model
                action, _ = model_select_move(model, env.board, env.current_player)
            else:  # minimax
                _, action = minimax(tuple(env.board), env.current_player)
            env.step(action)
        if env.winner == -1:  # minimax wins
            losses_x.append(first_move)
    
    if losses_x:
        print(f"\nModel loses as X when opening at: {losses_x}")
        for fm in losses_x:
            analyze_loss(model, model_plays_as=1, first_move=fm)
    else:
        print("\nModel never loses as X!")
    
    # Analyze Model=O losses (minimax opens, model responds)
    print("\n" + "="*70)
    print("DETAILED ANALYSIS: Model=O losses")
    print("="*70)
    
    losses_o = []
    for first_move in range(9):
        env = TicTacToeEnv()
        env.reset()
        env.step(first_move)  # minimax's opening
        while not env.done:
            if env.current_player == -1:  # model
                action, _ = model_select_move(model, env.board, env.current_player)
            else:  # minimax
                _, action = minimax(tuple(env.board), env.current_player)
            env.step(action)
        if env.winner == 1:  # minimax wins
            losses_o.append(first_move)
    
    if losses_o:
        print(f"\nModel loses as O when minimax opens at: {losses_o}")
        for fm in losses_o:
            analyze_loss(model, model_plays_as=-1, first_move=fm)
    else:
        print("\nModel never loses as O!")


if __name__ == "__main__":
    main()
