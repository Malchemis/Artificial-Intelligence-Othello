import random
import numpy as np

from minmax_params import TABLE1, TABLE2, MAX_INT, Strategy

from next import generate_moves, make_move
from bitwise_func import cell_count

from heuristics import positional, mobility, absolute

from visualize import cv2_display

MAX_DEPTH = 20


def strategy(mode: tuple, white_pieces: int, black_pieces: int, moves: list, turn: int, display: bool,
             size: int, max_depth: int) -> tuple:
    """Return the next move based on the strategy.

    Args:
        mode (tuple): describe the strategy and the player type.
        white_pieces (int): a bit board of the white pieces.
        black_pieces (int): a bit board of the black pieces.
        moves (list): list of possible moves.
        turn (int): current player.
        display (bool): display the board for the bots.
        size (int): size of the board.
        max_depth (int): max depth of the search.

    Returns:
        tuple: next move
    """
    global MAX_DEPTH
    MAX_DEPTH = max_depth

    if display:
        cv2_display(size, white_pieces, black_pieces, moves, turn, display_only=True)

    player = mode[0] if turn == -1 else mode[1]

    if player == Strategy.HUMAN:
        return s_human(white_pieces, black_pieces, moves, turn, size)

    if player == Strategy.RANDOM:
        return random.choice(moves)

    if player == Strategy.POSITIONAL_TABLE1:
        return negamax_alpha_beta(white_pieces, black_pieces, turn, 0, size, -MAX_INT, MAX_INT,
                                  heuristic=positional, table=TABLE1)[1]
    if player == Strategy.POSITIONAL_TABLE2:
        return negamax_alpha_beta(white_pieces, black_pieces, turn, 0, size, -MAX_INT, MAX_INT,
                                  heuristic=positional, table=TABLE2)[1]

    if player == Strategy.ABSOLUTE:
        return negamax_alpha_beta(white_pieces, black_pieces, turn, 0, size, -MAX_INT, MAX_INT,
                                  heuristic=absolute)[1]

    if player == Strategy.MOBILITY:
        return negamax_alpha_beta(white_pieces, black_pieces, turn, 0, size, -MAX_INT, MAX_INT,
                                  heuristic=mobility)[1]

    if player == Strategy.MIXED_TABLE1:
        return s_mixed(white_pieces, black_pieces, turn, 0, size, -MAX_INT, MAX_INT, TABLE1)[1]
    if player == Strategy.MIXED_TABLE2:
        return s_mixed(white_pieces, black_pieces, turn, 0, size, -MAX_INT, MAX_INT, TABLE2)[1]


def s_human(white_pieces: int, black_pieces: int, moves: list, turn: int, size: int) -> tuple:
    """Display the board using cv2 and return a move from the user"""
    return cv2_display(size, white_pieces, black_pieces, moves, turn)


def negamax_alpha_beta(white_pieces: int, black_pieces: int, turn: int, depth: int, size: int, alpha: int,
                       beta: int, heuristic=None, table=None) -> tuple | list:
    """
    Looks at the number of pieces flipped (tries to maximize (player's pieces - opponent's pieces)). MinMax (NegaMax)
    algorithm with alpha-beta pruning.

    Args:
        white_pieces (int): a bit board of the white player
        black_pieces (int): a bit board of the black player
        turn (int): current player
        depth (int): depth of the search
        size (int): size of the board
        alpha (int): alpha value
        beta (int): beta value
        heuristic : function to use to evaluate a board
        table : table of values for the heuristic

    Returns:
        tuple: best score, best move
    """
    own, enemy = (white_pieces, black_pieces) if turn == 1 else (black_pieces, white_pieces)

    # End of the recursion : Max depth reached or no more possible moves
    if depth == MAX_DEPTH:
        return heuristic(own, enemy, size, table), None
    moves, directions = generate_moves(own, enemy, size)
    if not moves:
        return heuristic(own, enemy, size, table), None

    best = -MAX_INT
    best_moves = []
    for move in moves:
        own, enemy = make_move(own, enemy, move, directions, size)
        if turn == 1:  # white plays as own
            score = -negamax_alpha_beta(own, enemy, -turn, depth + 1, size, -beta, -alpha, heuristic, table)[0]
        else:  # black plays as own
            score = -negamax_alpha_beta(enemy, own, -turn, depth + 1, size, -beta, -alpha, heuristic, table)[0]

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


def s_mixed(white_pieces: int, black_pieces: int, turn: int, depth: int, size: int, alpha: int, beta: int,
            table: np.ndarray) -> tuple:
    """Return the best move using phases. First phase (0-20) we use positional, then mobility, then absolute (
    44-64). The idea is we get a positional advantage early, then we try to make it hard for the opponent to play,
    then we maximize the number of pieces flipped

    Args:
        white_pieces (int): a bit board of the white player
        black_pieces (int): a bit board of the black player
        turn (int): current player
        depth (int): depth of the search
        size (int): size of the board
        alpha (int): alpha value
        beta (int): beta value
        table (np.ndarray): table of values for the heuristic

    Returns:
        int: best score, best move
    """
    count = cell_count(white_pieces | black_pieces)
    if count < 28:
        return negamax_alpha_beta(white_pieces, black_pieces, turn, depth, size, alpha, beta, positional, table)
    if count < 48:
        return negamax_alpha_beta(white_pieces, black_pieces, turn, depth, size, alpha, beta, mobility)
    return negamax_alpha_beta(white_pieces, black_pieces, turn, depth, size, alpha, beta, absolute)
