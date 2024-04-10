from node import Node
from strategies import strategy
from utils.bitwise_func import set_state, cell_count, print_board, print_pieces
from utils.minmax_params import Strategy  # Enums for the strategies
from utils.visualize import cv2_display


def othello(mode: tuple, minimax_mode: tuple, max_depth: int, h_table: tuple, thresholds: tuple, size: int, display: bool, verbose: int, stats_path: str) -> tuple[int, int, int, int, Node]:
    """
    Handles the game logic of Othello. The game is played on a 8x8 board by default by two players, one with the black
    pieces (value -1) and one with the white pieces (value +1). The game starts with 2 black pieces and 2 white pieces
    in the center of the board.


    Args:
        mode (tuple): describe the strategy and the player type.
        minimax_mode (tuple): describe the minimax version.
        max_depth (int): max depth of the search.
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
    own_root = Node(None, own, enemy, -1, size)  # -1 for black, 1 for white
    enemy_root = Node(None, own, enemy, -1, size)

    nb_pieces_played = 4  # You can put this to 0 if you don't consider the starting pieces
    while True:
        if verbose == 2:
            status(own, enemy, size, own_root.turn, nb_pieces_played)

        # Generate the possible moves for the current player
        if not own_root.visited:
            own_root.expand()
        # Current player can't play
        if not own_root.moves:  # Verify if the other player can play
            own_root.invert()  # Swap players and turn
            enemy_root = enemy_root.add_other_child(own_root)
            enemy_root.expand()
            if not enemy_root.moves:
                break  # End the game loop : No one can play
            enemy_root, own_root = own_root, enemy_root
            continue  # Skip the current turn

        if display:  # Display the board using OpenCV
            cv2_display(size, own_root.own_pieces, own_root.enemy_pieces, own_root.moves, own_root.turn,
                        display_only=True)

        # Get the next game/node from the strategy
        own_root = strategy(own_root, mode, minimax_mode, max_depth, h_table, thresholds, display, verbose, stats_path, nb_pieces_played)

        # We remove unused nodes to save memory (Garbage Collector)
        if own_root.parent is not None:
            own_root.parent.children = [own_root]

        # Advance the tree for the other player
        enemy_root = enemy_root.add_other_child(own_root)
        enemy_root, own_root = own_root, enemy_root  # Swap
        nb_pieces_played += 1  # and update metrics

    return (get_winner(own_root.own_pieces, own_root.enemy_pieces, verbose, own_root.turn),
            own_root.own_pieces,
            own_root.enemy_pieces,
            nb_pieces_played,
            own_root)


def error_handling(minimax_mode: tuple, mode: tuple, h_table: tuple, size: int) -> int:
    """
    Check if the input parameters are correct

    Args:
        minimax_mode (tuple): describe the version to use for the minimax algorithm.
        mode (tuple): describe the strategy and the player type.
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
    if not all(0 <= h < 2 for h in h_table):
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


def get_winner(own_pieces: int, enemy_pieces: int, verbose: bool, turn: int) -> int:
    """Print the winner and return the code of the winner

    Args:
        own_pieces (int): the pieces of the current player
        enemy_pieces (int): the pieces of the other player
        verbose (bool): print or not the winner
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
