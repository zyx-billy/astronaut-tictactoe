"""
Terminal UI: play Tic-Tac-Toe against the trained model.
User is X (1), model is O (-1). User moves first.
"""
import argparse
import numpy as np
from ttt import (
    load_weights,
    TTTModel,
    TicTacToeEnv,
    _board_to_numpy,
    _normalize_board,
)

# ANSI terminal colors
R = "\033[0m"       # reset
CX = "\033[36m"     # cyan for human
CO = "\033[33m"     # yellow for computer
LAST = "\033[1m\033[47m"  # bold + white background for last-played cell
VERT = " \033[90m│\033[0m "  # vertical separator between boards

HUMAN_SYM = "🚀"   # human (X)
COMP_SYM = "👨‍🚀"   # computer (O)

# ── Cell layout ──────────────────────────────────────────
# Both emojis are 2 terminal columns wide (regardless of Python str length).
# Each cell is 8 terminal columns: "   X    " for digits (3+1+4),
# "   E   " for emojis (3+2+3).
CELL_TCOLS = 8                          # terminal columns per cell
ROW_TCOLS = 3 * CELL_TCOLS + 2         # 26: 3 cells + 2 "|" separators

# Pre-built cell content (visible part only, no ANSI).
# The trick: each emoji is 2 terminal cols but different Python str lengths,
# so we hardcode the surrounding spaces per symbol.
HUMAN_CELL = f"   {HUMAN_SYM}   "      # 3 + emoji(2tcols) + 3 = 8 tcols
COMP_CELL  = f"   {COMP_SYM}   "       # 3 + emoji(2tcols) + 3 = 8 tcols

def _digit_cell(n: int) -> str:
    return f"   {n}    "                # 3 + 1 + 4 = 8 tcols

def _qval_cell(q: float) -> str:
    return f" {q:+.2f}  "              # 1 + 5 + 2 = 8 tcols

# Prediction grid row: " " + cell + " | " + cell + " | " + cell
PRED_ROW_TCOLS = 1 + 3 * CELL_TCOLS + 2 * 3  # 31


def main():
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe against the trained model.")
    parser.add_argument("weights", nargs="?", default="ttt_weights.bin",
                        help="Path to the model weights file (default: ttt_weights.bin)")
    args = parser.parse_args()
    weights_path = args.weights
    print("Loading model from", weights_path, "...")
    w1, w2, w3, b1, b2, b3 = load_weights(weights_path)
    model = TTTModel([w1, w2, w3, b1, b2, b3])
    print(f"Model loaded. You are {HUMAN_SYM}, the model is {COMP_SYM}. You move first.\n")

    env = TicTacToeEnv()
    env.reset()

    def _piece_cell(cell_val: int) -> str:
        """Return the plain (no ANSI) cell string for a placed piece."""
        return HUMAN_CELL if cell_val == 1 else COMP_CELL

    def game_board_lines(last_played_idx: int | None = None) -> list[str]:
        """Build game board rows. Every row is exactly ROW_TCOLS terminal columns."""
        lines = []
        for row in range(3):
            cells = []
            for col in range(3):
                idx = row * 3 + col
                cell = env.board[idx]
                is_last = last_played_idx is not None and idx == last_played_idx
                if cell == 0:
                    raw = _digit_cell(idx + 1)
                    cells.append(f"{LAST}{raw}{R}" if is_last else raw)
                else:
                    raw = _piece_cell(cell)
                    color = CX if cell == 1 else CO
                    if is_last:
                        cells.append(f"{LAST}{color}{raw}{R}")
                    else:
                        cells.append(f"{color}{raw}{R}")
            lines.append("|".join(cells))
            if row < 2:
                lines.append("-" * ROW_TCOLS)
        return lines

    def show_board(last_played_idx: int | None = None):
        print("\n".join(game_board_lines(last_played_idx)))

    def prediction_lines(board: list, player: int) -> list[str]:
        """Build prediction grid rows. Every data row is PRED_ROW_TCOLS terminal columns."""
        state_norm = _normalize_board(board, player)
        state_np = _board_to_numpy(state_norm)
        prediction = model.execute(state_np)
        display = prediction.copy()
        for i in range(9):
            if board[i] != 0:
                display[i] = np.nan
        who = HUMAN_SYM if player == 1 else COMP_SYM
        lines = [f"Predicted reward (Q) before that play ({who}'s view):"]
        for row in range(3):
            cells = []
            for col in range(3):
                idx = row * 3 + col
                if board[idx] != 0:
                    cells.append(_piece_cell(board[idx]))
                else:
                    cells.append(_qval_cell(display[idx]))
            row_str = " " + " | ".join(cells)
            lines.append(row_str)
            if row < 2:
                lines.append("-" * PRED_ROW_TCOLS)
        return lines

    def show_board_with_predictions(board_before: list, player_who_moved: int, last_played_idx: int):
        """Print game board (left) and prediction grid (right) side by side."""
        left_lines = game_board_lines(last_played_idx)
        right_lines = prediction_lines(board_before, player_who_moved)
        # Title row — ASCII only so len() == terminal cols; pad to ROW_TCOLS.
        left_title = " Game board" + " " * (ROW_TCOLS - len(" Game board"))
        left_all = [left_title] + left_lines
        right_all = [right_lines[0]] + right_lines[1:]
        # Ensure same length
        while len(left_all) < len(right_all):
            left_all.append(" " * ROW_TCOLS)
        while len(right_all) < len(left_all):
            right_all.append("")
        # Every left line is exactly ROW_TCOLS terminal columns, so just concat.
        sep = VERT
        for left, right in zip(left_all, right_all):
            print(left + sep + right)

    def model_move() -> int:
        state_norm = _normalize_board(env.board, env.current_player)
        state_np = _board_to_numpy(state_norm)
        prediction = model.execute(state_np)
        env.mask_illegal_moves(prediction)
        return int(np.argmax(prediction))

    show_board()  # initial board before any play
    while not env.done:
        if env.current_player == 1:
            legal = env.legal_moves()
            prompt = f"Your move (1–9, legal: {[m + 1 for m in legal]}): "
            while True:
                try:
                    raw = input(prompt).strip()
                    cell = int(raw)
                    if 1 <= cell <= 9:
                        action = cell - 1
                        if action in legal:
                            break
                except ValueError:
                    pass
                print("Invalid move. Try again.")
            print("You played", action + 1)
        else:
            action = model_move()
            print("Model plays", action + 1)

        board_before = env.board.copy()
        player_who_moved = env.current_player
        env.step(action)

        show_board_with_predictions(board_before, player_who_moved, last_played_idx=action)
        print("\n" + "─" * 60 + "\n")

    if env.winner == 1:
        print("You win!")
    elif env.winner == -1:
        print("Model wins!")
    else:
        print("Draw.")


if __name__ == "__main__":
    main()
