import cv2
import numpy as np

def cv2_display(size: int, board: np.ndarray, moves: set,  adj_cells: set, height: int = 800, width: int = 800, background: tuple = (0, 130, 0), display_only: bool = False, last_display: bool = False) -> tuple:
    """
    Display the Othello board using OpenCV

    Args:
        size (int): Size of the board
        board (np.ndarray): Board state of shape (size, size)
        moves (set): Set of possible moves
        height (int, optional): Height of the window. Defaults to 800.
        width (int, optional): Width of the window. Defaults to 800.
        background (tuple, optional): Background color. Defaults to (0, 110, 0).
        adj_cells (set, optional): Set of adjacent cells. Defaults to None.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    moves_x_y_only = [(move[0], move[1]) for move in list(moves)]
    
    # Set the background color
    img[:] = background
    # Change the background color for the adjacent cells
    if adj_cells is not None:
        for cell in adj_cells:
            cv2.rectangle(img, (cell[1]*100, cell[0]*100), (cell[1]*100+100, cell[0]*100+100), (0, 160, 0), -1)
    
    # Add pieces (circles)
    for i in range(size):
        for j in range(size):
            if board[i][j] == 1:
                cv2.circle(img, (j*100+50, i*100+50), 40, (255, 255, 255), -1)
            elif board[i][j] == -1:
                cv2.circle(img, (j*100+50, i*100+50), 40, (0, 0, 0), -1)
    # Add possible moves in grey (circles)
    for move in moves_x_y_only:
        cv2.circle(img, (move[1]*100+50, move[0]*100+50), 40, (128, 128, 128), -1)
    # Add grid lines
    for i in range(0, size):
        cv2.line(img, (0, i*100), (height, i*100), (0, 0, 0), 1)
        cv2.line(img, (i*100, 0), (i*100, width), (0, 0, 0), 1)     
    
    cv2.imshow("Othello", img)
    key = cv2.waitKey(1) &  0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit()
        
    if display_only:
        if last_display:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return
    # Wait for the user to click on a cell
    x, y = cv2_setMouseCallback(size, img)
    while (x, y) not in moves_x_y_only:
        x, y = cv2_setMouseCallback(size, img)
    cv2.destroyAllWindows()
    for move in moves:
        if move[0] == x and move[1] == y:
            return move
    return None

def cv2_setMouseCallback(size: int, img: np.ndarray) -> tuple:
    """
    Set the mouse callback for the OpenCV window

    Args:
        size (int): Size of the board
        img (np.ndarray): Board state of shape (size, size)

    Returns:
        tuple: x, y coordinates of the cell clicked
    """
    x = -1
    y = -1
    def mouse_callback(event, x_, y_, flags, param):
        nonlocal x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            # Representation is transposed of the board
            y = x_//100
            x = y_//100
    cv2.setMouseCallback("Othello", mouse_callback)
    while True:
        cv2.imshow("Othello", img)
        if x != -1 and y != -1:
            break
        cv2.waitKey(1)
    return x, y