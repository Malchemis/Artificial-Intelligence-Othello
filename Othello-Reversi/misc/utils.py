import numpy as np

BOARD_SIZE = 8


def set_state(bitboard: int, x: int, y: int, size: int):
    """Add a bit to the board by shifting a 1 to the left by x * size + y (flattened coordinates)"""
    return bitboard | (1 << (x * size + y))


def cell_count(bitboard: int):
    """Count the number of cells in the board"""
    return bitboard.bit_count()


def bits(number):
    """Generator to get the bits of a number"""
    bit = 1
    while number >= bit:
        if number & bit:
            yield bit
        bit <<= 1


def get_state(bitboard: int, x: int, y: int, size: int):
    """Return the state of the cell by shifting the board to the right by x * size + y and
    taking the least significant bit"""
    return (bitboard >> (x * size + y)) & 1


def get_indexes_move(move: int, size: int):
    """Return i and j indexes of the bitboard for a possible move"""
    position = move.bit_length() - 1
    # Calculate row and column indexes from the position
    i = position // size
    j = position % size
    return i, j


def print_pieces(bitboard: int, size: int):
    """Print the bit values of the board as a matrix of 0 and 1"""
    for i in range(size):
        for j in range(size):
            print(get_state(bitboard, i, j, size), end=' ')
        print()
    print()


def print_board(white: int, black: int, size: int):
    """Print the board with W for white, B for black and . for empty cells"""
    for i in range(size):
        for j in range(size):
            if get_state(white, i, j, size):
                print('W', end=' ')
            elif get_state(black, i, j, size):
                print('B', end=' ')
            else:
                print('.', end=' ')
        print()
    print()


def generate_moves(own, enemy, size) -> tuple[list, dict]:
    """Generate the possible moves for the current player using bitwise operations"""
    empty = ~(own | enemy)  # Empty squares (not owned by either player)
    unique_moves = []  # List of possible moves
    dir_jump = {}  # Dictionary of moves and the number of pieces that can be captured in each direction

    # Generate moves in all eight directions
    for direction in [north, south, east, west, north_west, north_east, south_west, south_east]:
        # We get the pieces that are next to an enemy piece in the direction
        count = 0
        victims = direction(own) & enemy
        if not victims:
            continue

        # We keep getting the pieces that are next to an enemy piece in the direction
        for _ in range(size):
            count += 1
            next_piece = direction(victims) & enemy
            if not next_piece:
                break
            victims |= next_piece

        # We get the pieces that can be captured in the direction
        captures = direction(victims) & empty
        # if there are multiple pieces in captures, we separate them and add them to the set
        while captures:
            capture = captures & -captures  # get the least significant bit
            captures ^= capture  # remove the lsb
            if capture not in dir_jump:
                unique_moves.append(capture)
                dir_jump[capture] = []
            dir_jump[capture].append((direction, count))

    return unique_moves, dir_jump


def make_move(own: int, enemy: int, move_to_play: int, directions: dict) -> tuple[int, int]:
    """Make the move and update the board using bitwise operations."""
    for direction, count in directions[move_to_play]:
        victims = move_to_play  # Init the victims with the move to play

        op_dir = opposite_dir(direction)  # opposite direction since we go from the move to play to the captured pieces
        for _ in range(count):
            victims |= (op_dir(victims) & enemy)
        own ^= victims
        enemy ^= victims & ~move_to_play
    # because of the XOR, the move to play which is considered a victim can be returned a pair number of times
    own |= move_to_play
    return own, enemy


# ------------------------------------ DIRECTIONS ------------------------------------ #
# Orthogonal directions
def north(x):
    return x >> 8


def south(x):
    return (x & 0x00ffffffffffffff) << 8


def east(x):
    return (x & 0x7f7f7f7f7f7f7f7f) << 1


def west(x):
    return (x & 0xfefefefefefefefe) >> 1


# Diagonal directions
def north_west(x):
    return north(west(x))


def north_east(x):
    return north(east(x))


def south_west(x):
    return south(west(x))


def south_east(x):
    return south(east(x))


opposite_direction = {north: south, south: north, east: west, west: east, north_west: south_east,
                      north_east: south_west,
                      south_west: north_east, south_east: north_west}


def opposite_dir(direction):
    return opposite_direction[direction]


import cv2


class Node:
    def __init__(self, parent, own_pieces, enemy_pieces, turn, size):
        self.parent = parent
        self.own_pieces = own_pieces
        self.enemy_pieces = enemy_pieces
        self.turn = turn
        self.size = size
        self.children = []
        self.moves = []
        self.directions = {}
        self.value = None
        self.visited = False

    def expand(self):
        self.moves, self.directions = generate_moves(self.own_pieces, self.enemy_pieces, self.size)
        self.visited = True

    def set_child(self, move):
        own, enemy = make_move(self.own_pieces, self.enemy_pieces, move, self.directions)
        child = Node(self, enemy, own, -self.turn, self.size)
        self.children.append(child)
        return child

    def set_children(self):
        for move in self.moves:
            own, enemy = make_move(self.own_pieces, self.enemy_pieces, move, self.directions)
            self.children.append(Node(self, enemy, own, -self.turn, self.size))

    def get_child(self, child):
        return self.children[self.children.index(child)]

    def add_other_child(self, other):
        child = Node(self, other.own_pieces, other.enemy_pieces, other.turn, other.size)
        self.children.append(child)
        return child

    def add_other_child_from_pieces(self, own, enemy):
        child = Node(self, own, enemy, -self.turn, self.size)
        self.children.append(child)
        return child

    def invert(self):
        self.enemy_pieces, self.own_pieces, self.turn = self.own_pieces, self.enemy_pieces, -self.turn

    def __eq__(self, other):
        return (self.own_pieces == other.own_pieces and self.enemy_pieces == other.enemy_pieces
                and self.turn == other.turn)

    def __str__(self):
        return f"{self.own_pieces}, {self.enemy_pieces}, {self.turn}"


HEIGHT = 600
WIDTH = 600
BACKGROUND_COLOR = (0, 130, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (125, 125, 125)


def init():
    enemy = set_state(0, BOARD_SIZE // 2 - 1, BOARD_SIZE // 2 - 1, BOARD_SIZE)
    enemy = set_state(enemy, BOARD_SIZE // 2, BOARD_SIZE // 2, BOARD_SIZE)
    own = set_state(0, BOARD_SIZE // 2 - 1, BOARD_SIZE // 2, BOARD_SIZE)
    own = set_state(own, BOARD_SIZE // 2, BOARD_SIZE // 2 - 1, BOARD_SIZE)
    return own, enemy


def cv2_display(size: int, own: int, enemy: int, moves: list, turn: int,
                display_only: bool = False, last_display: bool = False) -> int | None:
    """
    Display the Othello board using OpenCV

    Args:
        size (int): Size of the board.
        own (int): a bit board of the current player
        enemy (int): a bit board of the opponent
        moves (list): List of possible moves
        turn (int): Current turn (-1 for black, 1 for white)
        display_only (bool, optional): Flag to indicate if only display is required. Defaults to False.
        last_display (bool, optional): Flag to indicate if this is the last display. Defaults to False.

    Returns:
        int: The int corresponding to the selected move coordinates (x, y)
    """
    white, black = (own, enemy) if turn == 1 else (enemy, own)

    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # Set the background color
    img[:] = BACKGROUND_COLOR

    # Add pieces (circles)
    for i in range(size):
        for j in range(size):
            color = WHITE if (1 << (i * size + j)) & white else BLACK if (1 << (i * size + j)) & black else None
            if color is not None:
                cv2.circle(img, (j * WIDTH // size + WIDTH // (2 * size), i * HEIGHT // size + HEIGHT // (2 * size)),
                           int(min(WIDTH, HEIGHT) / (2.25 * size)), color, -1)

    # Add possible moves in grey, with a white or black border depending on the turn
    for move in moves:
        x, y = get_indexes_move(move, size)
        color = WHITE if turn == 1 else BLACK
        cv2.circle(img, (y * WIDTH // size + WIDTH // (2 * size), x * HEIGHT // size + HEIGHT // (2 * size)),
                   int(min(WIDTH, HEIGHT) / (2.25 * size)), color, 2)
        cv2.circle(img, (y * WIDTH // size + WIDTH // (2 * size), x * HEIGHT // size + HEIGHT // (2 * size)),
                   int(min(WIDTH, HEIGHT) / (2.25 * size)), GREY, -1)

    # Add grid lines
    for i in range(0, size):
        cv2.line(img, (0, i * HEIGHT // size), (WIDTH, i * HEIGHT // size), BLACK, 1)
        cv2.line(img, (i * WIDTH // size, 0), (i * WIDTH // size, HEIGHT), BLACK, 1)

    img = cv2.resize(img, (WIDTH, HEIGHT))
    cv2.imshow("Othello", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit(0)

    if display_only:
        if last_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
        return None

    # Wait for the user to click on a cell
    x, y = cv2_set_mouse_callback(img, size)
    while True:  # Verify if the cell is a valid move
        for move in moves:
            if (x, y) == get_indexes_move(move, size):
                return move
        x, y = cv2_set_mouse_callback(img, size)


def cv2_set_mouse_callback(img: np.ndarray, size: int) -> tuple:
    """
    Set the mouse callback for the OpenCV window

    Args:
        img (np.ndarray): Board state of shape (size, size)
        size (int): Size of the board.

    Returns:
        tuple: x, y coordinates of the cell clicked
    """
    x = -1
    y = -1

    def mouse_callback(event: int, x_: int, y_: int, flags: int, param: any) -> None:
        nonlocal x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            x = y_ * size // HEIGHT
            y = x_ * size // WIDTH

    cv2.setMouseCallback("Othello", mouse_callback, None)
    while True:
        cv2.imshow("Othello", img)
        if x != -1 and y != -1:
            break
        cv2.waitKey(1)
    return x, y


def replay(node: Node, size: int, display=False) -> list[Node]:
    """Replay the game based on the moves of the Node by backtracking the tree and using a LIFO queue"""
    game = []
    count_child = 0
    while node.parent is not None:
        game.append(node)
        node = node.parent
        count_child += len(node.children)
    game.append(node)
    game.reverse()
    if display:
        for node in game:
            if not node.moves:
                node.moves = generate_moves(node.own_pieces, node.enemy_pieces, size)[0]
            print("-------------")
            print(node)
            print(node.moves)
            print(node.value)
            cv2_display(size, node.own_pieces, node.enemy_pieces, node.moves, node.turn, display_only=True)
            print("Press Enter to continue...", end="")
            input()
        cv2.destroyAllWindows()
    return game
