from typing import Tuple

import numpy as np

DIRECTIONS = {(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)}


def play(board: np.ndarray, move: tuple, directions: list, turn: int, adjacent_cells: set, size: int) -> None:
    """Play the move

    Args:
        board (np.ndarray): board state
        move (dict): move  / key x, y of the dictionary of moves
        directions (list): list of directions (values of the dictionary of moves)
        turn (int): current player
        adjacent_cells (set): set of adjacent cells
        size (int): size of the board
    """
    old_x, old_y = move
    x, y = old_x, old_y
    board[x][y] = turn


    for n_jump, dx, dy in directions:
        for _ in range(n_jump):
            x += dx
            y += dy
            board[x][y] = turn
        x, y = old_x, old_y

    # update adjacent cells
    adjacent_cells.discard((old_x, old_y))
    for dx, dy in DIRECTIONS:
        if 0 <= old_x + dx < size and 0 <= old_y + dy < size:
            if board[old_x + dx][old_y + dy] == 0:
                adjacent_cells.add((old_x + dx, old_y + dy))


def get_possible_moves(board: np.ndarray, adjacent_cells: set, turn: int, size: int) -> dict:
    """
    Get the possible moves of the current player

    Args:
        board (np.ndarray): board state
        adjacent_cells (set): set of adjacent cells
        turn (int): current player
        size (int): size of the board

    Returns:
        list: list of possible moves
        set: set of invalid directions
    """
    possible_moves = dict()
    is_xy_valid = 0
    for x, y in adjacent_cells:
        possible_moves[(x, y)] = []
        for dx, dy in DIRECTIONS:
            is_valid, n_jump = is_valid_direction(board, x, y, dx, dy, turn, size)
            if is_valid:
                is_xy_valid = 1
                possible_moves[(x, y)].append((n_jump, dx, dy))
        if is_xy_valid == 0:
            del possible_moves[(x, y)]
        else:
            is_xy_valid = 0

    return possible_moves


def is_valid_direction(board: np.ndarray, x: int, y: int, dx: int, dy: int, turn: int, size: int) -> Tuple[bool, int]:
    """Check if the direction is valid, also return the last cell of the direction

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
    if x < 0 or x >= size or y < 0 or y >= size or board[x][y] == turn or board[x][y] == 0:
        return False, 0
    n_jump = 0
    max_jump = min(size - 1 - max(x, y), min(x, y))
    for _ in range(max_jump):
        if board[x][y] != -turn:
            break
        x += dx
        y += dy
        n_jump += 1
    return 0 <= x < size and 0 <= y < size and board[x][y] == turn and n_jump > 0, n_jump
