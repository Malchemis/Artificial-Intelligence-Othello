import cv2
import numpy as np
import time
import random

class Othello:
    def __init__(self, size=8, verbose=False):
        if size % 2 != 0:
            raise ValueError("Size must be an even number")
        self.size = size
        self.verbose = verbose
        self.init_board(size) # set the starting positions
        self.white_possible_moves = set([]) # white moves will be updated after the first move of black
        self.black_possible_moves = set([(size//2-1 -1, size//2-1), (size//2-1, size//2-1 -1), (size//2 +1, size//2), (size//2, size//2 +1)]) # black moves will be updated after the first move of white
    
    def init_board(self, size):
        self.turn = -1 # -1 for black,  1 for white
        self.board = np.zeros((size, size), dtype=int)
        self.board[size//2-1][size//2-1] =  1
        self.board[size//2][size//2] =  1
        self.board[size//2-1][size//2] = -1
        self.board[size//2][size//2-1] = -1
        # empty cells that are adjacent to the occupied cells
        self.adj_cells =   [(size//2-2, size//2-1), (size//2-1, size//2-2), (size//2-2, size//2-2), # top left
                            (size//2+1, size//2), (size//2, size//2+1), (size//2+1, size//2+1),     # bottom right
                            (size//2-2, size//2), (size//2-2, size//2+1), (size//2, size//2+1),     # top right
                            (size//2+1, size//2-1), (size//2, size//2-2), (size//2+1, size//2-2)]   # bottom left
    
    def get_possible_moves(self, color):
        # We check if an adjacent cell can be connected to a cell of the current color
        possible_moves = set()
        for x, y in self.adj_cells:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if self.is_valid_direction(x, y, dx, dy, color):
                        possible_moves.add((x, y))
                        break
        return possible_moves
    
    def is_valid_direction(self, x, y, dx, dy, color):
        x += dx
        y += dy
        while x >= 0 and x < self.size and y >= 0 and y < self.size and self.board[x][y] == -color:
            x += dx
            y += dy
        return x >= 0 and x < self.size and y >= 0 and y < self.size and self.board[x][y] == color
    
    def is_valid_move(self, x, y):
        return (x, y) in self.white_possible_moves if self.turn == 1 else (x, y) in self.black_possible_moves
    
    def make_move(self, x, y):
        if not self.is_valid_move(x, y):
            print(f'No possible move at ({x}, {y})') if self.verbose else None
            return False
        self.board[x][y] = self.turn
        self.update_adj_cells(x, y)
        self.flip_pieces(x, y)
        self.turn *= -1
        self.white_possible_moves = self.get_possible_moves(1)
        self.black_possible_moves = self.get_possible_moves(-1)
        return True
    
    def update_adj_cells(self, x, y):
        print(self.adj_cells) if self.verbose else None
        self.adj_cells.remove((x, y))
        print(self.adj_cells) if self.verbose else None
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if x+dx >= 0 and x+dx < self.size and y+dy >= 0 and y+dy < self.size and self.board[x+dx][y+dy] == 0:
                    self.adj_cells.append((x+dx, y+dy))

    def flip_pieces(self, x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if self.is_valid_direction(x, y, dx, dy, self.turn):
                    self.flip(x, y, dx, dy)
    
    def flip(self, x, y, dx, dy):
        x += dx
        y += dy
        while self.board[x][y] != self.turn:
            self.board[x][y] = self.turn
            x += dx
            y += dy
            
    def is_game_over(self):
        return len(self.white_possible_moves) == 0 and len(self.black_possible_moves) == 0
    
    def get_winner(self):
        white_count = np.sum(self.board == 1)
        black_count = np.sum(self.board == -1)
        if white_count > black_count:
            return 1
        if black_count > white_count:
            return -1
        return 0
    
    def cell_to_str(self, cell, x, y):
        if cell == 1:
            return "1"
        if cell == -1:
            return "-1"
        if (x, y) in self.white_possible_moves and self.turn == 1:
            return "#"
        if (x, y) in self.black_possible_moves and self.turn == -1:
            return "#"
        if (x, y) in self.adj_cells:
            return "X"
        return '0'
    
    def __str__(self):
        header = "------------OTHELLO------------\n"
        footer = "\n-------------------------------"
        cell_width =  3  # Set a fixed width for each cell
        formatted_rows = []
        for y, row in enumerate(self.board):
            formatted_row = " ".join(f"{self.cell_to_str(cell, x, y):>{cell_width}}" for x, cell in enumerate(row))
            formatted_rows.append(formatted_row)
        return header + "\n".join(formatted_rows) + footer
    
    def cv2_display(self, height=800, width=800, background=(0, 110, 0)):
        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cell_x = x // (width // self.size)
                cell_y = y // (height // self.size)
                if self.make_move(cell_y, cell_x):
                    print(othello)
                    cv2.imshow("Othello", self.get_display_image(height, width, background))

        cv2.namedWindow("Othello")
        cv2.setMouseCallback("Othello", on_click)

        while True:
            cv2.imshow("Othello", self.get_display_image(height, width, background))
            cv2.moveWindow("Othello", 1920, 0)  # Adjust the position according to your second screen

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_display_image(self, height, width, background):
        img = np.zeros((self.size, self.size, 3), dtype="uint8")
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    img[i][j] = background
                elif self.board[i][j] == 1:
                    img[i][j] = [255, 255, 255]
                elif self.board[i][j] == -1:
                    img[i][j] = [0, 0, 0]

        cell_width = width // self.size
        cell_height = height // self.size
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_NEAREST)
        for i in range(0, self.size):
            cv2.line(img, (0, i * cell_height), (height, i * cell_height), (0, 0, 0), 1)
            cv2.line(img, (i * cell_width, 0), (i * cell_width, width), (0, 0, 0), 1)

        return img



if __name__ == "__main__":
    othello = Othello(verbose=True)
    print(othello)
    othello.cv2_display()
    while True:
        print(f"Turn: {othello.turn}")
        # get possible moves
        moves = othello.white_possible_moves if othello.turn == 1 else othello.black_possible_moves
        print("Possible moves:", moves)
        if not moves:
            if othello.is_game_over():
                print("Game over")
                print("Winner:", othello.get_winner())
                break
            else:
                print("No possible moves, skipping turn")
                othello.turn *= -1
                continue
        # make a random move
        x, y = random.choice(list(moves))
        if othello.make_move(x, y):
            print(othello)
            # destroy the window and display the new board
            cv2.destroyAllWindows()
            othello.cv2_display()
        else:
            # refresh the display every dt
            othello.cv2_display()
            print("Invalid move")
        print()
        if othello.is_game_over():
            print("Game over")
            print("Winner:", othello.get_winner())
            break
    if cv2.getWindowProperty("Othello", cv2.WND_PROP_VISIBLE) != 0:
        cv2.destroyAllWindows()
