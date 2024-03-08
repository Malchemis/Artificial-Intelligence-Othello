import cv2
import numpy as np
from bitwise_func import get_indexes_move

HEIGHT = 600
WIDTH = 600
BACKGROUND_COLOR = (0, 130, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (125, 125, 125)


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
                exit(0)
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
