import random

from heuristics import positional, mobility, absolute
from minmax_params import TABLE1, TABLE2, MAX_INT, Strategy
from next import generate_moves, make_move
from visualize import cv2_display

MAX_DEPTH = 0


def strategy(minimax_mode: tuple, mode: tuple, own_pieces: int, enemy_pieces: int, moves: list, turn: int,
             display: bool, size: int, max_depth: int, save_moves: bool, nb_pieces_played) -> int:
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
        nb_pieces_played (int): number of pieces played.

    Returns:
        tuple: next move
    """
    global MAX_DEPTH
    MAX_DEPTH = max_depth

    if display:  # Display the board using OpenCV
        cv2_display(size, own_pieces, enemy_pieces, moves, turn, display_only=True)

    # Get the player type and the minimax version
    player, minimax_func = which_mode(mode, minimax_mode, turn)

    # Human player
    if player == Strategy.HUMAN:
        return s_human(own_pieces, enemy_pieces, moves, turn, size)

    # Random player
    if player == Strategy.RANDOM:
        return random.choice(moves)

    func_to_use = which_minimax(minimax_func)  # Get the minimax function to use

    # Define which table to use, table 1 or 2, or we don't care (for absolute, mobility and mixed) as it won't be used
    table_to_use = TABLE1 if player == Strategy.POSITIONAL_TABLE1 or player == Strategy.MIXED_TABLE1 else TABLE2

    # Define which heuristic to use
    heuristic_to_use = which_heuristic(player, nb_pieces_played)

    return func_to_use(own_pieces, enemy_pieces, turn, 0, size, -MAX_INT, MAX_INT,
                       heuristic=heuristic_to_use, table=table_to_use, save_moves=save_moves)[1]


def which_mode(mode: tuple, minimax_mode: tuple, turn: int) -> tuple:
    """Return the player type and the minimax version to use based on the turn"""
    if turn == 1:
        player = mode[0]
        minimax_func = minimax_mode[0]
    else:
        player = mode[1]
        minimax_func = minimax_mode[1]
    return player, minimax_func


def which_minimax(minimax_func: int) -> callable:
    """Return the minimax function to use based on the minimax version"""
    func_to_use = minimax  # default is minimax
    if minimax_func == Strategy.ALPHABETA:
        func_to_use = minimax_alpha_beta
    elif minimax_func == Strategy.NEGAMAX:
        func_to_use = negamax
    elif minimax_func == Strategy.NEGAMAX_ALPHA_BETA:
        func_to_use = negamax_alpha_beta
    return func_to_use


def which_heuristic(player: int, nb_pieces_played) -> callable:
    """Return the heuristic function to use based on the player type"""
    if player == Strategy.POSITIONAL_TABLE1 or player == Strategy.POSITIONAL_TABLE2:
        heuristic_to_use = positional
    elif player == Strategy.ABSOLUTE:
        heuristic_to_use = absolute
    elif player == Strategy.MOBILITY:
        heuristic_to_use = mobility
    else:
        heuristic_to_use = mixed_heuristic(nb_pieces_played)
    return heuristic_to_use


def mixed_heuristic(nb_pieces_played) -> callable:
    """Return the heuristic function to use based on the number of pieces played"""
    if nb_pieces_played < 15:
        return positional
    if nb_pieces_played < 45:
        return mobility
    return absolute


def s_human(own_pieces: int, enemy_pieces: int, moves: list, turn: int, size: int) -> int:
    """Display the board using cv2 and return a move from the user"""
    return cv2_display(size, own_pieces, enemy_pieces, moves, turn)


def minimax(own_pieces: int, enemy_pieces: int, turn: int, depth: int, size: int, alpha: int, beta: int,
            heuristic, table=None, save_moves=None) -> tuple:
    """MiniMax Algorithm"""
    # End of the recursion : Max depth reached or no more possible moves
    if depth == MAX_DEPTH:
        return heuristic(own_pieces, enemy_pieces, size, table), None
    moves, directions = generate_moves(own_pieces, enemy_pieces, size)
    if not moves:
        return heuristic(own_pieces, enemy_pieces, size, table), None

    best = -MAX_INT if depth % 2 == 0 else MAX_INT
    best_move = []
    for move in moves:
        new_enemy_pieces, new_own_pieces = make_move(own_pieces, enemy_pieces, move, directions)  # play and swap
        score = minimax(new_own_pieces, new_enemy_pieces, -turn, depth + 1, size, alpha, beta, heuristic, table,
                        save_moves)[0]

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
                       heuristic, table=None, save_moves=None) -> tuple:
    """MinMax Algorithm with alpha-beta pruning"""
    # End of the recursion : Max depth reached or no more possible moves
    if depth == MAX_DEPTH:
        return heuristic(own_pieces, enemy_pieces, size, table), None
    moves, directions = generate_moves(own_pieces, enemy_pieces, size)
    if not moves:
        return heuristic(own_pieces, enemy_pieces, size, table), None

    best = -MAX_INT if depth % 2 == 0 else MAX_INT
    best_move = []
    for move in moves:
        # Compute next move and score
        new_enemy_pieces, new_own_pieces = make_move(own_pieces, enemy_pieces, move, directions)  # play and swap
        score = minimax_alpha_beta(new_own_pieces, new_enemy_pieces, -turn, depth + 1, size, alpha, beta, heuristic,
                                   table, save_moves)[0]

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
            heuristic, table=None, save_moves=None) -> tuple:
    """Negamax version of the MinMax Algorithm"""
    # End of the recursion : Max depth reached or no more possible moves
    if depth == MAX_DEPTH:
        return heuristic(own_pieces, enemy_pieces, size, table), None
    moves, directions = generate_moves(own_pieces, enemy_pieces, size)
    if not moves:
        return heuristic(own_pieces, enemy_pieces, size, table), None

    best = -MAX_INT
    best_move = []
    for move in moves:
        new_enemy_pieces, new_own_pieces = make_move(own_pieces, enemy_pieces, move, directions)  # play and swap
        score = -negamax(new_own_pieces, new_enemy_pieces, -turn, depth + 1, size, alpha, beta, heuristic, table,
                         save_moves)[0]

        if score > best:
            best = score
            best_move = [move]
        elif score == best:
            best_move.append(move)
    return best, random.choice(best_move)


def negamax_alpha_beta(own_pieces: int, enemy_pieces: int, turn: int, depth: int, size: int, alpha: int, beta: int,
                       heuristic, table=None, save_moves=None) -> tuple:
    """Negamax version of the MinMax Algorithm with alpha-beta pruning. Only works for pair depth."""
    # End of the recursion : Max depth reached or no more possible moves
    if depth == MAX_DEPTH:
        return heuristic(own_pieces, enemy_pieces, size, table), None
    moves, directions = generate_moves(own_pieces, enemy_pieces, size)
    if not moves:
        return heuristic(own_pieces, enemy_pieces, size, table), None

    best = -MAX_INT
    best_move = []
    for move in moves:
        new_enemy_pieces, new_own_pieces = make_move(own_pieces, enemy_pieces, move, directions)  # play and swap
        score = -negamax_alpha_beta(new_own_pieces, new_enemy_pieces, -turn, depth + 1, size, -beta, -alpha, heuristic,
                                    table, save_moves)[0]

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
