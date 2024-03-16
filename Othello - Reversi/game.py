import yaml

from bitwise_func import set_state, cell_count, print_board, print_pieces
from measure import time_n  # Time measurement and Function Calls/Time Profiling
from minmax_params import Strategy  # Enums for the strategies
from strategies import strategy
from visualize import cv2_display
from next import generate_moves
from Node import Node


def othello(minimax_mode: tuple, mode: tuple, size: int = 8, max_depth: int = 4,
            display: bool = False, verbose: bool = False) -> tuple[int, int, int, int, Node]:
    """
    Handles the game logic of Othello. The game is played on a 8x8 board by default by two players, one with the black
    pieces (value -1) and one with the white pieces (value +1). The game starts with 2 black pieces and 2 white pieces
    in the center of the board.


    Args:
        minimax_mode (tuple): describe the strategy and the player type. First element is Black, and second is White.
        mode (tuple): describe the strategy and the player type. First element is Black, and second is White.
        size (int, optional): size of the board. Defaults to 8.
        max_depth (int, optional): max depth of the search tree. Defaults to 4.
        display (bool, optional): display the board for the bots. Defaults to False.
        verbose (bool, optional): print the winner. Defaults to False.

    Returns:
        tuple[int, int, int]: return code, white pieces, black pieces
    """
    error_handling(minimax_mode, mode, size)
    enemy, own = init_bit_board(size)  # set the bit board : white pieces, black pieces
    turn = -1  # Black starts

    own_root = Node(None, own, enemy, turn, size)
    enemy_root = Node(None, own, enemy, turn, size)

    nb_pieces_played = 4
    while True:
        if own_root != enemy_root:
            raise ValueError("Roots are not the same")

        if verbose == 2:
            status(own, enemy, size, turn, nb_pieces_played)

        # Generate the possible moves for the current player
        if not own_root.visited:
            own_root.expand()
        if len(own_root.moves) == 0:  # Verify if the other player can play
            if not enemy_root.visited:
                # duplicate the current node
                own_root = own_root.add_other_child_from_pieces(own_root.enemy_pieces, own_root.own_pieces)
                enemy_root = enemy_root.add_other_child(own_root)
                enemy_root.expand()
            if len(enemy_root.moves) == 0:
                break  # End the game loop : No one can play
            enemy_root, own_root = own_root, enemy_root
            turn *= -1
            continue  # Skip the current turn as the current player can't play

        if display:  # Display the board using OpenCV
            cv2_display(size, own_root.own_pieces, own_root.enemy_pieces, own_root.moves, turn, display_only=True)

        # Get the next Node following the strategy
        next_move = strategy(minimax_mode, mode, own_root, turn, size, max_depth, nb_pieces_played)
        own_root = own_root.set_child(next_move)

        # Advance the tree for the other player
        enemy_root = enemy_root.add_other_child(own_root)

        # Swap and update metrics
        enemy_root, own_root = own_root, enemy_root
        turn *= -1
        nb_pieces_played += 1

    return (get_winner(own_root.own_pieces, own_root.enemy_pieces, verbose, own_root.turn),
            own_root.own_pieces,
            own_root.enemy_pieces,
            nb_pieces_played,
            own_root)


def error_handling(minimax_mode: tuple, mode: tuple, size: int) -> int:
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

    if not all(Strategy.HUMAN <= m <= Strategy.MIXED_TABLE2 for m in mode):
        raise NotImplementedError("Invalid mode")
    if not all(Strategy.MINIMAX <= m <= Strategy.NEGAMAX_ALPHA_BETA for m in minimax_mode):
        raise NotImplementedError("Invalid minimax mode")

    if size != 8 and any(mode) in [Strategy.POSITIONAL_TABLE1, Strategy.POSITIONAL_TABLE2, Strategy.MIXED_TABLE1,
                                   Strategy.MIXED_TABLE2]:
        raise ValueError("Size must be 8 to use heuristic tables (TABLE1, TABLE2 are used by {2, 3, 6, 7})")
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


def replay(node: Node, size: int) -> None:
    """Replay the game based on the moves of the Node by backtracking the tree and using a LIFO queue"""
    game = []
    while node.parent is not None:
        game.append(node)
        node = node.parent
    game.reverse()
    for node in game:
        if not node.moves:
            node.moves = generate_moves(node.own_pieces, node.enemy_pieces, size)[0]
        cv2_display(size, node.own_pieces, node.enemy_pieces, node.moves, node.turn, display_only=True)
        print("Press Enter to continue...", end="")
        input()


def main():
    with open(f"{__file__}/../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    wins, onsets, offsets, nb_pieces_played_sum, nodes = time_n(
        othello,
        config["n"],
        (config["minimax_mode"], config["mode"], config["size"], config["max_depth"], config["display"],
         config["verbose"]),
        profile=config["profile"]
    )

    if config["replay"]:
        replay(nodes[0], config["size"])


if __name__ == "__main__":
    main()
