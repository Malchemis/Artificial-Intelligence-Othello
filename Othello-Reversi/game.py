import os
import pandas as pd

from node import Node, replay
from strategies import strategy
from utils.bitwise_func import set_state, cell_count, print_board, print_pieces
from utils.minmax_params import Strategy  # Enums for the strategies
from utils.visualize import cv2_display


def othello(mode: tuple, minimax_mode: tuple, max_depth: tuple, h_table: tuple, thresholds: tuple, size: int,
            display: bool, verbose: int, stats_path: str) -> tuple[int, int, int, int, Node]:
    """
    Handles the game logic of Othello. The game is played on a 8x8 board by default by two players, one with the black
    pieces (value -1) and one with the white pieces (value +1). The game starts with 2 black pieces and 2 white pieces
    in the center of the board.


    Args:
        mode (tuple): describe the strategy and the player type.
        minimax_mode (tuple): describe the minimax version.
        max_depth (tuple): max depth of the search.
        h_table (tuple): heuristic table to use.
        thresholds (tuple): threshold for the mixed strategy.
        size (int): size of the board.
        display (bool): display the board.
        verbose (int): verbose level.
        stats_path (str): path to save the stats.

    Returns:
        tuple[int, int, int]: return code, white pieces, black pieces
    """
    error_handling(minimax_mode, mode, h_table, size)

    enemy, own = init_bit_board(size)  # set the bitboards : white pieces, black pieces
    current_player = Node(None, own, enemy, -1, size)  # -1 for black, 1 for white
    other_player = Node(None, own, enemy, -1, size)

    nb_pieces_played = 4  # You can put this to 0 if you don't consider the starting pieces
    nb_nodes_generated = ([], [])
    while True:
        if verbose == 2:
            status(own, enemy, size, current_player.turn, nb_pieces_played)

        # Generate the possible moves for the current player
        if not current_player.visited:
            current_player.expand()
        # Current player can't play
        if not current_player.moves:  # Verify if the other player can play
            current_player.invert()  # Swap players and turn
            other_player = other_player.add_other_child(current_player)
            other_player.expand()
            if not other_player.moves:
                break  # End the game loop : No one can play
            other_player, current_player = current_player, other_player
            continue  # Skip the current turn

        if display:  # Display the board using OpenCV
            cv2_display(size, current_player.own_pieces, current_player.enemy_pieces, current_player.moves,
                        current_player.turn,
                        display_only=True)

        # Get the next game/node from the strategy
        current_player = strategy(current_player, mode, minimax_mode, max_depth, h_table, thresholds, nb_pieces_played)
        if stats_path:
            if current_player.turn == -1:
                nb_nodes_generated[0].append(current_player.count_all_children())
            else:
                nb_nodes_generated[1].append(current_player.count_all_children())

        # Advance the tree for the other player
        other_player = other_player.add_other_child(current_player)
        collect(current_player, other_player)  # Remove parent's children to save memory
        other_player, current_player = current_player, other_player  # Swap
        nb_pieces_played += 1  # and update metrics

    if stats_path:
        save_stats(current_player, nb_pieces_played, nb_nodes_generated, stats_path, mode, minimax_mode, max_depth,
                   h_table, thresholds)
    return (get_winner(current_player.own_pieces, current_player.enemy_pieces, verbose, current_player.turn),
            current_player.own_pieces,
            current_player.enemy_pieces,
            nb_pieces_played,
            current_player)


def error_handling(minimax_mode: tuple, mode: tuple, h_table: tuple, size: int) -> int:
    """
    Check if the input parameters are correct

    Args:
        minimax_mode (tuple): describe the version to use for the minimax algorithm.
        mode (tuple): describe the strategy and the player type.
        h_table (tuple): heuristic table to use.
        size (int): size of the board
    """
    if size < 4:
        raise ValueError("Size must be at least 4")
    if size % 2 != 0:
        raise ValueError("Size must be an even number")

    if not all(Strategy.HUMAN <= m <= Strategy.MIXED for m in mode):
        raise NotImplementedError("Invalid mode")
    if not all(Strategy.MINIMAX <= m <= Strategy.NEGAMAX_ALPHA_BETA for m in minimax_mode):
        raise NotImplementedError("Invalid minimax mode")
    if not all(0 <= h <= 2 for h in h_table):
        raise NotImplementedError("Invalid heuristic table")

    if size != 8:
        raise NotImplementedError("Only 8x8 board is supported for now")
    return 0


def init_bit_board(size) -> tuple[int, int]:
    """Set the starting positions for the white and black pieces"""
    white_pieces = set_state(0, size // 2 - 1, size // 2 - 1, size)
    white_pieces = set_state(white_pieces, size // 2, size // 2, size)
    black_pieces = set_state(0, size // 2 - 1, size // 2, size)
    black_pieces = set_state(black_pieces, size // 2, size // 2 - 1, size)
    return white_pieces, black_pieces


def get_winner(own_pieces: int, enemy_pieces: int, verbose: int, turn: int) -> int:
    """Print the winner and return the code of the winner

    Args:
        own_pieces (int): the pieces of the current player
        enemy_pieces (int): the pieces of the other player
        verbose (int): print or not the winner
        turn (int): the current player
    
    Returns:
        int: return code. -1 if black wins, 0 if tied, 1 if white wins
    """
    white_pieces, black_pieces = (own_pieces, enemy_pieces) if turn == 1 else (enemy_pieces, own_pieces)
    black = cell_count(black_pieces)
    white = cell_count(white_pieces)
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


def status(own: int, enemy: int, size: int, turn: int, nb_pieces_played: int) -> None:
    str_status = "Black" if turn == -1 else "White"
    print(f"Turn: {str_status}, Pieces played: {nb_pieces_played}")
    white_pieces, black_pieces = (own, enemy) if turn == 1 else (enemy, own)
    print_board(white_pieces, black_pieces, size)
    print_pieces(white_pieces, size)
    print_pieces(black_pieces, size)
    print(f"{white_pieces:064b}")
    print(f"{black_pieces:064b}")
    print(white_pieces | black_pieces)


def collect(current_player: Node, other_player: Node) -> None:
    """Remove the children of the parent node to save memory"""
    if current_player.parent:
        del current_player.parent.children
        del current_player.parent.moves_to_child
        current_player.parent.visited = False
        current_player.parent.children = [current_player]
    if other_player.parent:
        del other_player.parent.children
        del other_player.parent.moves_to_child
        other_player.parent.visited = False
        other_player.parent.children = [other_player]


def save_stats(node: Node, nb_pieces_played: int, nb_nodes_generated: tuple, stats_path: str,
               mode: tuple, minimax_mode: tuple, max_depth: tuple, h_table: tuple, thresholds: tuple) \
        -> None:
    """Save the stats of the game.
    Append to the CSV file the following stats :
    - Who played as Black and White
    - The score of Black and White
    - The number of pieces played
    - A list of evaluated states for each player
    - The number of nodes generated for each player

    Args:
        node (Node): the last node of the game
        nb_pieces_played (int): the number of pieces played
        nb_nodes_generated (tuple): the number of nodes generated for each player (a tuple of lists)
        mode (tuple): describe the strategy and the player type.
        minimax_mode (tuple): describe the minimax version.
        max_depth (tuple): max depth of the search.
        h_table (tuple): heuristic table to use.
        thresholds (tuple): threshold for the mixed strategy.
        stats_path (str): the path to save the stats
    """
    # check who played as Black and White
    black_pieces, white_pieces = (node.own_pieces, node.enemy_pieces) if node.turn == -1 else (node.enemy_pieces,
                                                                                               node.own_pieces)
    black_score = cell_count(black_pieces)
    white_score = cell_count(white_pieces)

    # Backtrack the nodes to get the list of states
    game_states = replay(node, node.size, False)
    black_evaluated_states = []
    white_evaluated_states = []
    for state in game_states:
        if state.turn == -1:
            black_evaluated_states.append(state.value)
        else:
            white_evaluated_states.append(state.value)

    # Get the identity of the players from the parameters
    current = [mode[0], minimax_mode[0], max_depth[0], h_table[0], thresholds[0]]
    other = [mode[1], minimax_mode[1], max_depth[1], h_table[1], thresholds[1]]
    black_identity, white_identity = (current, other) if node.turn == -1 else (other, current)

    # Append the stats to the CSV file
    if not os.path.exists(stats_path):
        stats = pd.DataFrame(columns=["Black", "White", "Black score", "White score", "Pieces played",
                                      "Black evaluated states", "White evaluated states",
                                      "Number of nodes generated by Black", "Number of nodes generated by White"])
        stats.to_csv(stats_path, index=False)
    stats = pd.read_csv(stats_path)
    df = pd.DataFrame({"Black": [black_identity], "White": [white_identity], "Black score": [black_score],
                       "White score": [white_score], "Pieces played": [nb_pieces_played],
                       "Black evaluated states": [black_evaluated_states],
                       "White evaluated states": [white_evaluated_states],
                       "Number of nodes generated by Black": [nb_nodes_generated[0]],
                       "Number of nodes generated by White": [nb_nodes_generated[1]]})
    stats = pd.concat([stats, df], ignore_index=True, sort=False)
    stats.to_csv(stats_path, index=False)
