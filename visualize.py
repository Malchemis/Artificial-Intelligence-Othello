import cv2
import numpy as np

def cv2_display(size: int, board: np.ndarray, moves, height: int = 800, width: int = 800, background: tuple = (0, 110, 0)):
    """
    Display the Othello board using OpenCV

    Args:
        size (int): Size of the board
        board (np.ndarray): Board state of shape (size, size)
        moves (set): Set of possible moves
        height (int, optional): Height of the window. Defaults to 800.
        width (int, optional): Width of the window. Defaults to 800.
        background (tuple, optional): Background color. Defaults to (0, 110, 0).
    """
    img = np.zeros((size, size, 3), dtype="uint8")
    moves_x_y_only = [(move[0], move[1]) for move in list(moves)]
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0:
                img[i][j] = background
            if board[i][j] == 1:
                img[i][j] = [255, 255, 255]
            if board[i][j] == -1:
                img[i][j] = [0, 0, 0]

    img = cv2.resize(img, (height, width), interpolation=cv2.INTER_NEAREST)
    # Add grid lines
    for i in range(0, size):
        cv2.line(img, (0, i*100), (height, i*100), (0, 0, 0), 1)
        cv2.line(img, (i*100, 0), (i*100, width), (0, 0, 0), 1)     
    # Add possible moves in grey (circles)
    for move in moves_x_y_only:
        cv2.circle(img, (move[1]*100+50, move[0]*100+50), 30, (100, 100, 100), -1)
    cv2.imshow("Othello", img)
    key = cv2.waitKey(1) &  0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit()