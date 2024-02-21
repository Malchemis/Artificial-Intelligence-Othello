import random

import numpy as np

from minmax_params import TABLE1, TABLE2, MAX_DEPTH, MAX_INT
from next import get_possible_moves, play
from visualize import cv2_display


def strategy(mode: tuple, board: np.ndarray, moves: dict, turn: int, adjacent_cells: set, display: bool,
             size: int) -> tuple:
    """Return the next move based on the strategy.

    Args:
        mode (tuple): describe the strategy and the player type.
        board (np.ndarray): board state.
        moves (dict): list of possible moves.
        turn (int): current player.
        adjacent_cells (set): set of adjacent cells.
        display (bool): display the board for the bots.
        size (int): size of the board.

    Returns:
        tuple: next move
    """
    if display:
        cv2_display(size, board, moves, turn, adjacent_cells, display_only=True)

    if turn == -1:
        player = mode[0]
    else:
        player = mode[1]

    if player == 0:
        return s_human(board, moves, adjacent_cells, turn, size)
    if player == 1:
        return random.choice(list(moves.keys()))
    if player == 2:
        return s_positional(board, adjacent_cells, turn, 0, TABLE1, size, -MAX_INT, MAX_INT)[1]
    if player == 3:
        return s_positional(board, adjacent_cells, turn, 0, TABLE2, size, -MAX_INT, MAX_INT)[1]
    if player == 4:
        return s_absolute(board, adjacent_cells, turn, 0, size, -MAX_INT, MAX_INT)[1]
    if player == 5:
        return s_mobility(board, adjacent_cells, turn, 0, size, -MAX_INT, MAX_INT)[1]
    if player == 6:
        return s_mixed(board, adjacent_cells, turn, 0, TABLE1, size, -MAX_INT, MAX_INT)[1]
    if player == 7:
        return s_mixed(board, adjacent_cells, turn, 0, TABLE2, size, -MAX_INT, MAX_INT)[1]


def s_human(board: np.ndarray, moves: dict, adj_cells, turn, size: int) -> tuple:
    """Display the board using cv2 and return a move from the user

    Args:
        board (np.ndarray): board state
        moves (dict): Dict of possible moves
        adj_cells (set): set of adjacent cells
        turn (int): current player
        size (int): size of the board

    Returns:
        tuple: next move
    """
    return cv2_display(size, board, moves, turn, adj_cells)


def s_absolute(board: np.ndarray, adjacent_cells: set, turn: int, depth: int, size: int, alpha: int,
               beta: int) -> tuple | list:
    """
    Looks at the number of pieces flipped (tries to maximize (player's pieces - opponent's pieces)). MinMax (NegaMax)
    algorithm with alpha-beta pruning.

    Args:
        board (np.ndarray): board state
        adjacent_cells (set): set of adjacent cells
        turn (int): current player
        depth (int): depth of the search
        size (int): size of the board
        alpha (int): alpha value
        beta (int): beta value

    Returns:
        tuple: best score, best move
    """
    if depth == MAX_DEPTH or len(adjacent_cells) == 0:
        return [np.sum(board == turn) - np.sum(board == -turn)]
    moves = get_possible_moves(board, adjacent_cells, turn, size)
    if len(moves) == 0:
        return [np.sum(board == turn) - np.sum(board == -turn)]

    best = -MAX_INT
    best_moves = []
    for move in moves:
        board_copy = board.copy()
        adj_cells = adjacent_cells.copy()
        play(board_copy, move, moves[move], turn, adj_cells, size)
        score = -s_absolute(board_copy, adj_cells, -turn, depth + 1, size, -beta, -alpha)[0]
        if score == best:
            best_moves.append(move)
        if score > best:
            best = score
            best_moves = [move]
            if best > alpha:
                alpha = best
                if alpha >= beta:
                    break
    best_move = random.choice(best_moves)
    return best, best_move


def s_positional(board: np.ndarray, adjacent_cells: set, turn: int, depth: int, table: np.ndarray, size: int,
                 alpha: int, beta: int) -> tuple | list:
    """Return the best move using heuristics. MinMax (NegaMax) algorithm with alpha-beta pruning

    Args:
        board (np.ndarray): board state
        adjacent_cells (set): set of adjacent cells
        turn (int): current player
        depth (int): depth of the search
        table (np.ndarray): table of adjacent cells
        size (int): size of the board
        alpha (int): alpha value
        beta (int): beta value

    Returns:
        tuple: best score, best move
    """
    if depth == MAX_DEPTH or len(adjacent_cells) == 0:
        return [np.sum(table[np.where(board == turn)]) - np.sum(table[np.where(board == -turn)])]
    moves = get_possible_moves(board, adjacent_cells, turn, size)
    if len(moves) == 0:
        return [np.sum(table[np.where(board == turn)]) - np.sum(table[np.where(board == -turn)])]
    best = -MAX_INT
    best_moves = []
    for move in moves:
        board_copy = board.copy()
        adj_cells = adjacent_cells.copy()
        play(board_copy, move, moves[move], turn, adj_cells, size)
        score = -s_positional(board_copy, adj_cells, -turn, depth + 1, table, size, -beta, -alpha)[0]
        if score == best:
            best_moves.append(move)
        if score > best:
            best = score
            best_moves = [move]
            if best > alpha:
                alpha = best
                if alpha >= beta:
                    break
    best_move = random.choice(best_moves)
    return best, best_move


def s_mobility(board: np.ndarray, adjacent_cells: set, turn: int, depth: int, size: int, alpha: int,
               beta: int) -> tuple | list:
    """Return the best move using the mobility. Maximize the number of possible moves for the current player,
    and minimize the number of possible moves for the other player. MinMax (NegaMax) algorithm with alpha-beta pruning

    Args:
        board (np.ndarray): board state
        adjacent_cells (set): set of adjacent cells
        turn (int): current player
        depth (int): depth of the search
        size (int): size of the board
        alpha (int): alpha value
        beta (int): beta value

    Returns:
        int: best score, best move
    """
    if len(adjacent_cells) == 0:
        return [turn]
    moves = get_possible_moves(board, adjacent_cells, turn, size)
    length_moves = len(moves)
    if depth == MAX_DEPTH or length_moves == 0:
        return [length_moves - len(get_possible_moves(board, adjacent_cells, -turn, size))]

    best = -MAX_INT
    best_moves = []
    for move in moves:
        board_copy = board.copy()
        adj_cells = adjacent_cells.copy()
        play(board_copy, move, moves[move], turn, adj_cells, size)
        score = -s_mobility(board_copy, adj_cells, -turn, depth + 1, size, -beta, -alpha)[0]
        if score == best:
            best_moves.append(move)
        if score > best:
            best = score
            best_moves = [move]
            if best > alpha:
                alpha = best
                if alpha >= beta:
                    break
    best_move = random.choice(best_moves)
    return best, best_move


def s_mixed(board: np.ndarray, adjacent_cells: set, turn: int, depth: int, table: np.ndarray, size: int, alpha: int,
            beta: int) -> tuple:
    """Return the best move using phases. First phase (0-20) we use positional, then mobility, then absolute (
    44-64). The idea is we get a positional advantage early, then we try to make it hard for the opponent to play,
    then we maximize the number of pieces flipped

    Args:
        board (np.ndarray): board state
        adjacent_cells (set): set of adjacent cells
        turn (int): current player
        depth (int): depth of the search
        table (np.ndarray): table of values
        size (int): size of the board
        alpha (int): alpha value
        beta (int): beta value

    Returns:
        int: best score, best move
    """
    if np.sum(board != 0) < 25:
        return s_positional(board, adjacent_cells, turn, depth, table, size, alpha, beta)
    if np.sum(board != 0) < 45:
        return s_mobility(board, adjacent_cells, turn, depth, size, alpha, beta)
    return s_absolute(board, adjacent_cells, turn, depth, size, alpha, beta)
