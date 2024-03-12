import random

import numpy as np

from bitwise_func import cell_count
from heuristics import positional, mobility, absolute
from minmax_params import TABLE1, TABLE2, MAX_INT, Strategy
from next import get_next_moves, make_move
from visualize import cv2_display

MAX_DEPTH = 20


def strategy(minimax_mode: tuple, mode: tuple, own_pieces: int, enemy_pieces: int, moves: list, turn: int,
             display: bool, size: int, max_depth: int,
             save_moves: bool, own_knowledge: dict, count_level: int) -> int:
    """Return the next move based on the strategy.

    Args:
        minimax_mode (tuple): describe the minimax version.
        mode (tuple): describe the strategy and the player type.
        own_pieces (int): a bit board of the own pieces.
        enemy_pieces (int): a bit board of the enemy pieces.
        moves (list): list of possible moves.
        turn (int): current player.
        display (bool): display the board for the bots.
        size (int): size of the board.
        max_depth (int): max depth of the search.
        save_moves (bool): save the moves as knowledge for each player (separately).
        own_knowledge (dict): knowledge of the current player.
        count_level (int): number of pieces on the board / depth of the game.

    Returns:
        tuple: next move
    """
    global MAX_DEPTH
    MAX_DEPTH = max_depth

    if display:
        cv2_display(size, own_pieces, enemy_pieces, moves, turn, display_only=True)

    player = mode[0] if turn == -1 else mode[1]
    minimax_func = minimax_mode[0] if turn == -1 else minimax_mode[1]

    func_to_use = minimax  # default is minimax
    if minimax_func == Strategy.ALPHABETA:
        func_to_use = minimax_alpha_beta
    elif minimax_func == Strategy.NEGAMAX:
        func_to_use = negamax
    elif minimax_func == Strategy.NEGAMAX_ALPHA_BETA:
        func_to_use = negamax_alpha_beta

    if player == Strategy.HUMAN:
        return s_human(own_pieces, enemy_pieces, moves, turn, size)

    if player == Strategy.RANDOM:
        return random.choice(moves)

    # Define which table to use, table 1 or 2, or we don't care (for absolute, mobility and mixed) as it won't be used
    table_to_use = TABLE1 if player == Strategy.POSITIONAL_TABLE1 or player == Strategy.MIXED_TABLE1 else TABLE2

    if player == Strategy.MIXED_TABLE1 or player == Strategy.MIXED_TABLE2:
        return s_mixed(func_to_use, own_pieces, enemy_pieces, turn, 0, size, -MAX_INT, MAX_INT, table_to_use,
                       save_moves=save_moves, own_knowledge=own_knowledge, count_level=count_level)[1]

    # Define which heuristic to use
    if player == Strategy.POSITIONAL_TABLE1 or player == Strategy.POSITIONAL_TABLE2:
        heuristic_to_use = positional
    elif player == Strategy.ABSOLUTE:
        heuristic_to_use = absolute
    elif player == Strategy.MOBILITY:
        heuristic_to_use = mobility
    else:
        heuristic_to_use = None

    return func_to_use(own_pieces, enemy_pieces, turn, 0, size, -MAX_INT, MAX_INT,
                       heuristic=heuristic_to_use, table=table_to_use, save_moves=save_moves,
                       own_knowledge=own_knowledge, count_level=count_level)[1]


def s_human(own_pieces: int, enemy_pieces: int, moves: list, turn: int, size: int) -> int:
    """Display the board using cv2 and return a move from the user"""
    return cv2_display(size, own_pieces, enemy_pieces, moves, turn)


def minimax(own_pieces: int, enemy_pieces: int, turn: int, depth: int, size: int, alpha: int, beta: int,
            heuristic, table=None, save_moves=None, own_knowledge=None, count_level=None) -> tuple:
    """
    MinMax Algorithm
    Args:
        own_pieces (int): a bit board of the current player
        enemy_pieces (int): a bit board of the enemy player
        turn (int): current player
        depth (int): depth of the search
        size (int): size of the board
        alpha (int): not used. Only put here for compatibility/versatility in function calling in strategy()
        beta (int): not used. Only put here for compatibility/versatility in function calling in strategy()
        heuristic : function to use to evaluate a board
        table : table of values for the heuristic
        save_moves (bool): save the moves as knowledge for each player (separately)
        own_knowledge (dict): knowledge of the current player
        count_level (int): number of pieces on the board / depth of the game

    Returns:
        tuple: best score, best move
    """
    # End of the recursion : Max depth reached or no more possible moves
    if depth == MAX_DEPTH:
        return heuristic(own_pieces, enemy_pieces, size, table), None
    moves, directions = get_next_moves(own_pieces, enemy_pieces, size, save_moves, own_knowledge, count_level + depth)
    if not moves:
        return heuristic(own_pieces, enemy_pieces, size, table), None

    best = -MAX_INT if depth % 2 == 0 else MAX_INT
    best_move = []
    for move in moves:
        new_enemy_pieces, new_own_pieces = make_move(own_pieces, enemy_pieces, move, directions)  # play and swap
        score = minimax(new_own_pieces, new_enemy_pieces, -turn, depth + 1, size, alpha, beta, heuristic, table,
                        save_moves, own_knowledge, count_level + 1)[0]

        if depth % 2 == 0:
            if score > best:
                best = score
                best_move = [move]
            elif score == best:
                best_move.append(move)
        else:
            if score < best:
                best = score
                best_move = [move]
            elif score == best:
                best_move.append(move)
    return best, random.choice(best_move)


def minimax_alpha_beta(own_pieces: int, enemy_pieces: int, turn: int, depth: int, size: int, alpha: int, beta: int,
                       heuristic, table=None, save_moves=None, own_knowledge=None, count_level=None) -> tuple:
    """
    MinMax Algorithm with alpha-beta pruning
    Args:
        own_pieces (int): a bit board of the current player
        enemy_pieces (int): a bit board of the enemy player
        turn (int): current player
        depth (int): depth of the search
        size (int): size of the board
        alpha (int): alpha value
        beta (int): beta value
        heuristic : function to use to evaluate a board
        table : table of values for the heuristic
        save_moves (bool): save the moves as knowledge for each player (separately)
        own_knowledge (dict): knowledge of the current player
        count_level (int): number of pieces on the board / depth of the game

    Returns:
        tuple: best score, best move
    """
    # End of the recursion : Max depth reached or no more possible moves
    if depth == MAX_DEPTH:
        return heuristic(own_pieces, enemy_pieces, size, table), None
    moves, directions = get_next_moves(own_pieces, enemy_pieces, size, save_moves, own_knowledge, count_level + depth)
    if not moves:
        return heuristic(own_pieces, enemy_pieces, size, table), None

    best = -MAX_INT if depth % 2 == 0 else MAX_INT
    best_move = []
    for move in moves:
        # Compute next move and score
        new_enemy_pieces, new_own_pieces = make_move(own_pieces, enemy_pieces, move, directions)  # play and swap
        score = minimax_alpha_beta(new_own_pieces, new_enemy_pieces, -turn, depth + 1, size, alpha, beta, heuristic,
                                   table, save_moves, own_knowledge, count_level+1)[0]

        # Update best score and best move
        if score == best:
            best_move.append(move)
        else:
            if depth % 2 == 0:
                if score > best:
                    best = score
                    best_move = [move]
                alpha = max(alpha, best)  # Prune if possible
                if alpha >= beta:
                    break
            else:
                if score < best:
                    best = score
                    best_move = [move]
                beta = min(beta, best)  # Prune if possible
                if alpha >= beta:
                    break
    return best, random.choice(best_move)


def negamax(own_pieces: int, enemy_pieces: int, turn: int, depth: int, size: int, alpha: int, beta: int,
            heuristic, table=None, save_moves=None, own_knowledge=None, count_level=None) -> tuple:
    """
    Negamax version of the MinMax Algorithm

    Args:
        own_pieces (int): a bit board of the current player
        enemy_pieces (int): a bit board of the enemy player
        turn (int): current player
        depth (int): depth of the search
        size (int): size of the board
        alpha (int): not used. Only put here for compatibility/versatility in function calling in strategy()
        beta (int): not used. Only put here for compatibility/versatility in function calling in strategy()
        heuristic : function to use to evaluate a board
        table : table of values for the heuristic
        save_moves (bool): save the moves as knowledge for each player (separately)
        own_knowledge (dict): knowledge of the current player
        count_level (int): number of pieces on the board / depth of the game

    Returns:
        tuple: best score, best move
    """
    # End of the recursion : Max depth reached or no more possible moves
    if depth == MAX_DEPTH:
        return heuristic(own_pieces, enemy_pieces, size, table), None
    moves, directions = get_next_moves(own_pieces, enemy_pieces, size, save_moves, own_knowledge, count_level + depth)
    if not moves:
        return heuristic(own_pieces, enemy_pieces, size, table), None

    best = -MAX_INT
    best_move = []
    for move in moves:
        new_enemy_pieces, new_own_pieces = make_move(own_pieces, enemy_pieces, move, directions)  # play and swap
        score = -negamax(new_own_pieces, new_enemy_pieces, -turn, depth + 1, size, alpha, beta, heuristic, table,
                         save_moves, own_knowledge, count_level+1)[0]

        if score > best:
            best = score
            best_move = [move]
        elif score == best:
            best_move.append(move)
    return best, random.choice(best_move)


def negamax_alpha_beta(own_pieces: int, enemy_pieces: int, turn: int, depth: int, size: int, alpha: int, beta: int,
                       heuristic, table=None, save_moves=None, own_knowledge=None, count_level=None) -> tuple:
    """
    Negamax version of the MinMax Algorithm with alpha-beta pruning. Only works for pair depth.

    Args:
        own_pieces (int): a bit board of the current player
        enemy_pieces (int): a bit board of the enemy player
        turn (int): current player
        depth (int): depth of the search
        size (int): size of the board
        alpha (int): alpha value
        beta (int): beta value
        heuristic : function to use to evaluate a board
        table : table of values for the heuristic
        save_moves (bool): save the moves as knowledge for each player (separately)
        own_knowledge (dict): knowledge of the current player
        count_level (int): number of pieces on the board / depth of the game

    Returns:
        tuple: best score, best move
    """
    # End of the recursion : Max depth reached or no more possible moves
    if depth == MAX_DEPTH:
        return heuristic(own_pieces, enemy_pieces, size, table), None
    moves, directions = get_next_moves(own_pieces, enemy_pieces, size, save_moves, own_knowledge, count_level + depth)
    if not moves:
        return heuristic(own_pieces, enemy_pieces, size, table), None

    best = -MAX_INT
    best_move = []
    for move in moves:
        new_enemy_pieces, new_own_pieces = make_move(own_pieces, enemy_pieces, move, directions)  # play and swap
        score = -negamax_alpha_beta(new_own_pieces, new_enemy_pieces, -turn, depth + 1, size, -beta, -alpha, heuristic,
                                    table, save_moves, own_knowledge, count_level+1)[0]

        if score == best:
            best_move.append(move)
        elif score > best:
            best = score
            best_move = [move]
            if best > alpha:
                alpha = best
                if alpha > beta:
                    break
    return best, random.choice(best_move)


def s_mixed(minimax_function, own_pieces: int, enemy_pieces: int, turn: int, depth: int, size: int,
            alpha: int, beta: int, table: np.ndarray,
            phase_threshold=None, save_moves=None, own_knowledge=None, count_level=None) -> tuple:
    """Return the best move using phases. First phase (0-20) we use positional, then mobility, then absolute (
    44-64). The idea is we get a positional advantage early, then we try to make it hard for the opponent to play,
    then we maximize the number of pieces flipped

    Args:
        minimax_function (function): minimax function to use
        own_pieces (int): a bit board of the current player
        enemy_pieces (int): a bit board of the enemy player
        turn (int): current player
        depth (int): depth of the search
        size (int): size of the board
        alpha (int): alpha value
        beta (int): beta value
        table (np.ndarray): table of values for the heuristic
        phase_threshold (list, optional): threshold for the phases. Defaults to [28, 48].
        save_moves (bool): save the moves as knowledge for each player (separately)
        own_knowledge (dict): knowledge of the current player
        count_level (int): number of pieces on the board / depth of the game

    Returns:
        tuple: best score, best move
    """
    if phase_threshold is None or not (isinstance(phase_threshold, list) and len(phase_threshold) == 2):
        phase_threshold = [28, 48]
    count = cell_count(own_pieces | enemy_pieces)
    if count < phase_threshold[0]:
        return minimax_function(own_pieces, enemy_pieces, turn, depth, size, alpha, beta, positional, table,
                                save_moves, own_knowledge, count_level)
    if count < phase_threshold[1]:
        return minimax_function(own_pieces, enemy_pieces, turn, depth, size, alpha, beta, mobility, table,
                                save_moves, own_knowledge, count_level)
    return minimax_function(own_pieces, enemy_pieces, turn, depth, size, alpha, beta, absolute, table,
                            save_moves, own_knowledge, count_level)
