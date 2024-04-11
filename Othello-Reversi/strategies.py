import random

from heuristics import positional, mobility, absolute
from node import Node
from utils.minmax_params import TABLE1, TABLE2, MAX_INT, Strategy, Heuristic
from utils.visualize import cv2_display


def strategy(node: Node, mode: tuple, minimax_mode: tuple, max_depth: tuple, h_table: tuple, thresholds: tuple,
             nb_pieces_played: int) -> Node:
    """Return the next move based on the strategy.

    Args:
        node (Node): the root node of the search tree.
        mode (tuple): describe the strategy and the player type.
        minimax_mode (tuple): describe the minimax version.
        max_depth (tuple): max depth of the search. (tuple to allow different depths for each player)
        h_table (tuple): heuristic table to use.
        thresholds (tuple): threshold for the mixed strategy.
        nb_pieces_played (int): number of pieces played.
        
    Returns:
        Node: next board state.
    """
    # Get the player type, the minimax version, the heuristic table
    player, minimax_func, heuristic = which_mode(mode, minimax_mode, h_table, node.turn)

    # Human player
    if player == Strategy.HUMAN:
        next_move = s_human(node)
        return node.set_child(next_move)

    # Random player
    if player == Strategy.RANDOM:
        next_move = random.choice(node.moves)
        return node.set_child(next_move)

    # Any MiniMax Player
    func_to_use = which_minimax(minimax_func)  # Get the minimax function to use

    # Define which heuristic table to use, table 1 or 2, or we don't care (for absolute/mobility) as it won't be used
    table_to_use = TABLE1 if (heuristic == Heuristic.TABLE1) else TABLE2 if (heuristic == Heuristic.TABLE2) else None

    # Define which strategy to use
    heuristic_to_use = which_strategy(player, nb_pieces_played, thresholds)

    # Define the max depth to use
    max_depth_to_use = max_depth[0] if node.turn == -1 else max_depth[1]
    next_node = func_to_use(node, heuristic_to_use, max_depth=max_depth_to_use, table=table_to_use)
    return next_node


def which_mode(mode: tuple, minimax_mode: tuple, h_table, turn: int) -> tuple:
    """Return the player type and the minimax version to use based on the turn"""
    if turn == -1:
        player = mode[0]
        minimax_func = minimax_mode[0]
        heuristic = h_table[0]
    else:
        player = mode[1]
        minimax_func = minimax_mode[1]
        heuristic = h_table[1]
    return player, minimax_func, heuristic


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


def which_strategy(player: int, nb_pieces_played: int, thresholds: tuple) -> callable:
    """Return the heuristic function to use based on the player type"""
    if player == Strategy.POSITIONAL:
        heuristic_to_use = positional
    elif player == Strategy.ABSOLUTE:
        heuristic_to_use = absolute
    elif player == Strategy.MOBILITY:
        heuristic_to_use = mobility
    else:
        heuristic_to_use = mixed_heuristic(nb_pieces_played, thresholds)
    return heuristic_to_use


def mixed_heuristic(nb_pieces_played: int, thresholds: tuple) -> callable:
    """Return the heuristic function to use based on the number of pieces played"""
    if nb_pieces_played < thresholds[0]:
        return positional
    if nb_pieces_played < thresholds[1]:
        return mobility
    return absolute


def s_human(node: Node) -> int:
    """Display the board using cv2 and return a move from the user"""
    # Get the int corresponding to the selected move coordinates (x, y)
    move = cv2_display(node.size, node.own_pieces, node.enemy_pieces, node.moves, node.turn)
    return move


def minimax(node: Node, heuristic: callable, max_depth: int, depth: int = 0, table=None) -> Node:
    """MiniMax Algorithm"""
    # End of the recursion : Max depth reached or no more possible moves
    if depth == max_depth:
        node.value = heuristic(node.own_pieces, node.enemy_pieces, node.size, table)
        return node

    if not node.visited:
        node.expand()
    if not node.moves:
        node.value = heuristic(node.own_pieces, node.enemy_pieces, node.size, table)
        return node

    best = -MAX_INT if depth % 2 == 0 else MAX_INT
    best_nodes = []
    for move in node.moves:
        child = node.set_child(move)
        child.value = minimax(child, heuristic, max_depth, depth=depth + 1, table=table).value

        if depth % 2 == 0:
            if child.value > best:
                best = child.value
                best_nodes = [child]
            elif child.value == best:
                best_nodes.append(child)
        else:
            if child.value < best:
                best = child.value
                best_nodes = [child]
            elif child.value == best:
                best_nodes.append(child)
    return random.choice(best_nodes)


def minimax_alpha_beta(node: Node, heuristic: callable, max_depth: int, depth: int = 0,
                       alpha: int = -MAX_INT, beta: int = MAX_INT, table=None) -> Node:
    """MinMax Algorithm with alpha-beta pruning"""
    # End of the recursion : Max depth reached or no more possible moves
    if depth == max_depth:
        node.value = heuristic(node.own_pieces, node.enemy_pieces, node.size, table)
        return node

    if not node.visited:
        node.expand()
    if not node.moves:
        node.value = heuristic(node.own_pieces, node.enemy_pieces, node.size, table)
        return node

    best = -MAX_INT if depth % 2 == 0 else MAX_INT
    best_nodes = []
    for move in node.moves:
        child = node.set_child(move)
        child.value = minimax_alpha_beta(child, heuristic, max_depth,
                                         depth=depth + 1, alpha=alpha, beta=beta, table=table).value

        # Update best node and best score
        if child.value == best:
            best_nodes.append(child)
        else:
            if depth % 2 == 0:
                if child.value > best:
                    best = child.value
                    best_nodes = [child]
                alpha = max(alpha, best)  # Prune if possible
                if alpha >= beta:
                    return random.choice(best_nodes)
            else:
                if child.value < best:
                    best = child.value
                    best_nodes = [child]
                beta = min(beta, best)  # Prune if possible
                if alpha >= beta:
                    return random.choice(best_nodes)
    return random.choice(best_nodes)


def negamax(node: Node, heuristic: callable, max_depth: int, depth: int = 0, alpha: int = -MAX_INT,
            beta: int = MAX_INT, table=None) -> Node:
    """Negamax version of the MinMax Algorithm. Only works for pair depth."""
    # End of the recursion : Max depth reached or no more possible moves
    if depth == max_depth:
        node.value = heuristic(node.own_pieces, node.enemy_pieces, node.size, table)
        return node

    if not node.visited:
        node.expand()
    if not node.moves:
        node.value = heuristic(node.own_pieces, node.enemy_pieces, node.size, table)
        return node

    best = -MAX_INT
    best_nodes = []
    for move in node.moves:
        child = node.set_child(move)
        child.value = -negamax(child, heuristic, max_depth,
                               depth=depth + 1, alpha=-beta, beta=-alpha, table=table).value

        if child.value > best:
            best = child.value
            best_nodes = [child]
        elif child.value == best:
            best_nodes.append(child)
    return random.choice(best_nodes)


def negamax_alpha_beta(node: Node, heuristic: callable, max_depth: int, depth: int = 0,
                       alpha: int = -MAX_INT, beta: int = MAX_INT, table=None) -> Node:
    """Negamax version of the MinMax Algorithm with alpha-beta pruning. Only works for pair depth."""
    # End of the recursion : Max depth reached or no more possible moves
    if depth == max_depth:
        node.value = heuristic(node.own_pieces, node.enemy_pieces, node.size, table)
        return node

    if not node.visited:
        node.expand()
    if not node.moves:
        node.value = heuristic(node.own_pieces, node.enemy_pieces, node.size, table)
        return node

    best = -MAX_INT
    indexes = []
    for i, move in enumerate(node.moves):
        if node.moves_to_child[move] is None:
            node.set_child(move)
        child = node.moves_to_child[move]
        child.value = -negamax_alpha_beta(child, heuristic, max_depth,
                                          depth=depth + 1, alpha=-beta, beta=-alpha, table=table).value

        if child.value > best:
            best = child.value
            indexes = [i]
            if best > alpha:
                alpha = best
                if alpha > beta:
                    return node.children[random.choice(indexes)]
        elif child.value == best:
            indexes.append(i)
    return node.children[random.choice(indexes)]
