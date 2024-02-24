from typing import Tuple

import numpy as np

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]


def play(board: np.ndarray, move: tuple, directions: list, turn: int, adjacent_cells: set, size: int) -> None:
    """Play the move

    Args:
        board (np.ndarray): board state
        move (tuple): move  / key x, y of the dictionary of moves
        directions (list): list of directions (values of the dictionary of moves)
        turn (int): current player
        adjacent_cells (set): set of adjacent cells
        size (int): size of the board
    """
    x, y = move
    board[x][y] = turn

    for n_jump, dx, dy in directions:
        for _ in range(n_jump):
            x += dx
            y += dy
            board[x][y] = turn
        x, y = move

    # update adjacent cells
    adjacent_cells.discard(move)
    for dx, dy in DIRECTIONS:
        new_x, new_y = move[0] + dx, move[1] + dy
        if 0 <= new_x < size and 0 <= new_y < size and board[new_x][new_y] == 0:
            adjacent_cells.add((new_x, new_y))


def get_possible_moves(board: np.ndarray, adjacent_cells: set, turn: int, size: int) -> list:
    """
    Get the possible moves of the current player

    Args:
        board (np.ndarray): board state
        adjacent_cells (set): set of adjacent cells
        turn (int): current player
        size (int): size of the board

    Returns:
        list: list of possible moves with key (x, y) and value list of directions
    """
    possible_moves = []
    for x, y in adjacent_cells:
        directions = []
        for dx, dy in DIRECTIONS:
            is_valid, n_jump = is_valid_direction(board, x, y, dx, dy, turn, size)
            if is_valid:
                directions.append((n_jump, dx, dy))
        if directions:
            possible_moves.append(((x, y), directions))

    return possible_moves


def is_valid_direction(board: np.ndarray, x: int, y: int, dx: int, dy: int, turn: int, size: int) -> Tuple[bool, int]:
    """Check if the direction is valid, and return the number of jumps in the direction

    Args:
        board (np.ndarray): board state
        x (int): x coordinate
        y (int): y coordinate
        dx (int): x direction
        dy (int): y direction
        turn (int): current player
        size (int): size of the board

    Returns:
        bool: True if the direction is valid
        int: number of jumps in the direction
    """
    x += dx
    y += dy
    n_jump = 0
    while 0 <= x < size and 0 <= y < size and board[x][y] == -turn:
        x += dx
        y += dy
        n_jump += 1
    return 0 <= x < size and 0 <= y < size and board[x][y] == turn and n_jump > 0, n_jump
