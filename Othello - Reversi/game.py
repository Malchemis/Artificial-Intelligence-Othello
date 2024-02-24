from typing import Tuple

import numpy as np

from measure import time_n
from measure import profile_n
from next import get_possible_moves, play
from strategies import strategy


def othello(mode: tuple, size: int = 8, display: bool = False, verbose: bool = False) \
        -> Tuple[int, np.ndarray, list, set]:
    """
    Handles the game logic of Othello. We keep track of the board, the turn, the possible moves and the adjacent
    cells.
    - The game is played on a 8x8 board by default.
    - The game is played by two players, one with the black pieces (value -1) and one with the white pieces (value +1).
    Empty cells are represented by 0.
    - The game starts with 2 black pieces and 2 white pieces in the center of the board.

    The play mode is defined as follows:
    0 for Human, >=1 for Bot (1: random, 2: positional with TABLE1, 3: positional with TABLE2, 4: absolute, 5: mobility,
    6: mixed).

    Args:
        mode (tuple): describe the strategy and the player type. Index 0 is Black, and 1 is White.
        size (int, optional): size of the board. Defaults to 8.
        display (bool, optional): display the board for the bots. Defaults to False.
        verbose (bool, optional): print the winner. Defaults to False.

    Returns:
        int: return code. -1 if black wins, 0 if tied, 1 if white wins
    """
    error_handling(mode, size)
    board = np.zeros((size, size))
    adjacent_cells = set()

    turn = -1  # Black starts
    init_board(board)  # set the starting positions
    init_adjacent_cells(adjacent_cells)  # set the adjacent cells

    while True:
        moves = get_possible_moves(board, adjacent_cells, turn, size)  # set the possible moves
        if not moves:
            # verify if the other player can play
            if len(get_possible_moves(board, adjacent_cells, -turn, size)) == 0:
                break
            turn *= -1
            continue

        next_move = strategy(mode, board, moves, turn, adjacent_cells, display, size)
        play(board, next_move[0], next_move[1], turn, adjacent_cells, size)
        turn *= -1

    return get_winner(board, verbose), board, moves, adjacent_cells


def error_handling(mode: tuple, size: int) -> int:
    """
    Check if the input parameters are correct

    Args:
        mode (tuple): describe the strategy and the player type.
        size (int): size of the board
    """
    if size < 4:
        raise ValueError("Size must be at least 4")
    if size % 2 != 0:
        raise ValueError("Size must be an even number")
    if not all(0 <= m <= 7 for m in mode):
        raise ValueError("Invalid mode")
    return 0


def init_board(board: np.ndarray, size: int = 8) -> None:
    """
    Set the starting positions of the board

    Args:
        board (np.ndarray): board state
        size (int, optional): size of the board. Defaults to 8.
    """
    board[size // 2 - 1][size // 2 - 1] = 1
    board[size // 2][size // 2] = 1
    board[size // 2 - 1][size // 2] = -1
    board[size // 2][size // 2 - 1] = -1



def init_adjacent_cells(adjacent_cells: set, size: int = 8) -> None:
    """
    Set the adjacent cells

    Args:
        adjacent_cells (set): set of adjacent cells
        size (int, optional): size of the board. Defaults to 8.
    """
    adjacent_cells.update(
        [(size // 2 - 2, size // 2 - 1), (size // 2 - 1, size // 2 - 2), (size // 2 - 2, size // 2 - 2),  # top left
         (size // 2 + 1, size // 2), (size // 2, size // 2 + 1), (size // 2 + 1, size // 2 + 1),  # bottom right
         (size // 2 - 2, size // 2), (size // 2 - 2, size // 2 + 1), (size // 2 - 1, size // 2 + 1),  # bottom left
         (size // 2 + 1, size // 2 - 1), (size // 2, size // 2 - 2), (size // 2 + 1, size // 2 - 2)])  # top right


def get_winner(board: np.ndarray, verbose: bool) -> int:
    """Print the winner and return the code of the winner

    Args:
        board (np.ndarray): board state
        verbose (bool): print or not the winner
    
    Returns:
        int: return code. -1 if black wins, 0 if tied, 1 if white wins
    """
    black = np.sum(board == -1)
    white = np.sum(board == 1)
    if black > white:
        if verbose:
            print("Black wins" + "(" + str(black) + " vs " + str(white) + ")")
        return -1
    if black < white:
        if verbose:
            print("White wins" + "(" + str(white) + " vs " + str(black) + ")")
        return 1
    if verbose:
        print("Draw" + "(" + str(black) + " vs " + str(white) + ")")
    return 0


def main():
    # r_code, r_board, r_moves, r_adj_cells = othello((1, 1), 8, False, False)
    # cv2_display(8, r_board, r_moves, 1, r_adj_cells, display_only=True, last_display=True)
    time_n(othello, 1000, ((1, 1), 8, False, False))
    profile_n(othello, 1000, ((1, 1), 8, False, False))


if __name__ == "__main__":
    main()