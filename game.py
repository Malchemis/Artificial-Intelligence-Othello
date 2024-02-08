import numpy as np
import time
import pandas as pd
import os

import random
from visualize import cv2_display
from minimax_params import TABLE, MAX_DEPTH, MAX_INT


# INITIALIZATION for the CSV file
csv_columns = ['function', 'execution_time', 'intermediate_times'] # Execution time is the time it took to execute the function
path_dir = 'results'
file_name = 'execution_times.csv'
file2_name = 'intermediate_results.csv'
os.makedirs('results', exist_ok=True)
results = pd.DataFrame(columns=csv_columns)
intermediate_results = pd.DataFrame(columns=csv_columns[:-1]) # intermediate results for the execution time

def timit(func):
    """Decorator to time the function and write the execution time to a CSV file

    Args:
        func (function): function to be timed

    Returns:
        function: decorated function
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        execution_time = end - start
        # Sum the execution time, and add to the list of intermediate time
        intermediate_results.loc[len(intermediate_results)] = [func.__name__, execution_time]        
        return result
    return wrapper

# @timit
def othello(mode:int, strategy_i:int=None, size:int = 8, display:bool = False) -> int:
    """Handles the game logic of Othello. We keep track of the board, the turn, the possible moves and the adjacent cells.
    - The game is played on an 8x8 board by default.
    - The game is played by two players, one with the black pieces (value -1) and one with the white pieces (value +1). Empty cells are represented by 0.
    - The game starts with 2 black pieces and 2 white pieces in the center of the board.
    - We keep track of the player's turn 
    

    Args:
        mode (int): describe if it's BotvBot, BotvHuman, HumanvBot or HumanvHuman (0, 1, 2, 3 respectively)
        strategy_i (int, optional): describe the strategy of the bot. Defaults to None (~random), 0: greedy, 1: minimax
        size (int, optional): size of the board. Defaults to 8.
        display (bool, optional): display the board for the bots. Defaults to False.
    Returns:
        int: return code. 0 if finished correclty
    """
    error_handling(mode, strategy_i, size)
    # init board, turn, adjacent cells, possible moves
    board = np.zeros((size, size), dtype=int)
    adjacent_cells = set()
    turn = -1 # Black starts
    init_board(board)                   # set the starting positions
    init_adjacent_cells(adjacent_cells) # set the adjacent cells
    while len(adjacent_cells) > 0:
        moves, invalid_directions = get_possible_moves(board, adjacent_cells, turn) # set the possible moves
        if len(moves) == 0:
            # verify if the other player can play
            if len(get_possible_moves(board, adjacent_cells, -turn)[0]) == 0:
                break
            turn *= -1
            continue
        copy_turn = turn # we are careful not to modify the turn, specially for minimax
        next_move = strategy(mode, strategy_i, board, moves, copy_turn, adjacent_cells.copy(), display=display)
        play(board, next_move, turn, adjacent_cells, invalid_directions) # flip the cells, update adjacent cells, update possible moves
        turn *= -1
    get_winner(board)
    return 0, board, moves, adjacent_cells

# @timit
def error_handling(mode:int, strategy_i:int, size:int) -> None:
    """Check if the input parameters are correct

    Args:
        mode (int): describe if it's BotvBot, BotvHuman, HumanvBot or HumanvHuman (0, 1, 2, 3 respectively)
        strategy_i (int): describe the strategy of the bot. None: random, 0: greedy, 1: minimax
        size (int): size of the board
    """
    if size < 4:
        raise ValueError("Size must be at least 4")
    if size % 2 != 0:
        raise ValueError("Size must be an even number")
    if mode not in [0, 1, 2, 3]:
        raise NotImplementedError("Mode not implemented")
    if strategy_i not in [None, 0, 1]:
        raise NotImplementedError("Strategy not implemented")

# @timit
def init_board(board: np.ndarray, size:int = 8) -> None:
    """Set the starting positions of the board

    Args:
        board (np.ndarray): board state
        size (int, optional): size of the board. Defaults to 8.
    """
    board[size//2-1][size//2-1] =  1
    board[size//2][size//2] =  1
    board[size//2-1][size//2] = -1
    board[size//2][size//2-1] = -1
    
# @timit
def init_adjacent_cells(adjacent_cells: set, size:int = 8) -> None:
    """Set the adjacent cells

    Args:
        adjacent_cells (set): set of adjacent cells
        size (int, optional): size of the board. Defaults to 8.
    """
    adjacent_cells.update( [(size//2-2, size//2-1), (size//2-1, size//2-2), (size//2-2, size//2-2), # top left
                            (size//2+1, size//2), (size//2, size//2+1), (size//2+1, size//2+1),     # bottom right
                            (size//2-2, size//2), (size//2-2, size//2+1), (size//2-1, size//2+1),   # bottom left
                            (size//2+1, size//2-1), (size//2, size//2-2), (size//2+1, size//2-2)])  # bottom left


# @timit
def get_possible_moves(board: np.ndarray, adjacent_cells: set, turn: int) -> set:
    """Get the possible moves of the current player

    Args:
        board (np.ndarray): board state
        adjacent_cells (set): set of adjacent cells
        turn (int): current player

    Returns:
        set: set of possible moves
    """
    possible_moves = set()
    invalid_directions = set()
    for x, y in adjacent_cells:
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            is_valid, n_jump = is_valid_direction(board, x, y, dx, dy, turn)
            if is_valid:
                possible_moves.add((x, y, n_jump, dx, dy))
                break
            else:
                invalid_directions.add((dx, dy))
        invalid_directions = set()
    return possible_moves, invalid_directions

# @timit
def is_valid_direction(board: np.ndarray, x: int, y: int, dx: int, dy: int, turn: int) -> bool:
    """Check if the direction is valid, also return the last cell of the direction

    Args:
        board (np.ndarray): board state
        x (int): x coordinate
        y (int): y coordinate
        dx (int): x direction
        dy (int): y direction
        turn (int): current player

    Returns:
        bool: True if the direction is valid
    """
    x += dx
    y += dy
    n_jump = 0
    for _ in range(board.shape[0]):
        if x < 0 or x >= board.shape[0] or y < 0 or y >= board.shape[1] or board[x][y] != -turn:
            break
        x += dx
        y += dy
        n_jump += 1
    return x >= 0 and x < board.shape[0] and y >= 0 and y < board.shape[1] and board[x][y] == turn and n_jump > 0, n_jump

# @timit
def strategy(mode:int, strategy_i:int, board: np.ndarray, moves: set, turn: int, adj_cells, display=False) -> tuple:
    """Return the next move

    Args:
        mode (int): describe if it's BotvBot, BotvHuman, HumanvBot or HumanvHuman (0, 1, 2, 3 respectively)
        strategy_i (int): describe the strategy of the bot. None: random, 0: greedy, 1: minimax
        board (np.ndarray): board state
        moves (set): set of possible moves
        turn (int): current player

    Returns:
        tuple: next move
    """
    if mode == 0 or (mode == 1 and turn == -1) or (mode == 2 and turn == 1):
        return strategy_bot(strategy_i, board, moves, turn, adj_cells, display=display)
    else:
        return strategy_human(board, moves, adj_cells)

# @timit
def play(board: np.ndarray, move: tuple, turn: int, adjacent_cells: set, invalid_directions: set) -> None:
    """Play the move

    Args:
        board (np.ndarray): board state
        move (tuple): next move
        turn (int): current player
    """
    x, y, n_jump, dx, dy = move
    old_x, old_y = x, y
    board[x][y] = turn
    # We update the direction we know for sure is valid
    for _ in range(n_jump):
        x += dx
        y += dy
        board[x][y] = turn
    x, y = old_x, old_y
    # We update other possible directions (the break in the get_possible_moves function prevent us from having more than 1 valid direction)
    for dx, dy in set([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]) - invalid_directions:
        is_valid, n_jump = is_valid_direction(board, x, y, dx, dy, turn)
        if is_valid:
            for _ in range(n_jump):
                x += dx
                y += dy
                board[x][y] = turn
            x, y = old_x, old_y
    
    # update adjacent cells
    adjacent_cells.discard((old_x, old_y))
    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        if old_x+dx >= 0 and old_x+dx < board.shape[0] and old_y+dy >= 0 and old_y+dy < board.shape[1] and board[old_x+dx][old_y+dy] == 0:
            adjacent_cells.add((old_x+dx, old_y+dy))

def strategy_bot(strategy_i:int, board: np.ndarray, moves: set, turn: int, adj_cells, display=False) -> tuple:
    """Return the next move

    Args:
        strategy_i (int): describe the strategy of the bot. None: random, 0: greedy, 1: minimax
        board (np.ndarray): board state
        moves (set): set of possible moves
        turn (int): current player

    Returns:
        tuple: next move
    """
    if display:
        cv2_display(board.shape[0], board, moves, adj_cells, display_only=True)
    if strategy_i is None:
        return random.choice(list(moves))
    if strategy_i == 0:
        return greedy(board, moves, turn)
    return negamax(board, adj_cells, turn, strategy_i)[1]

def strategy_human(board: np.ndarray, moves: set, adj_cells) -> tuple:
    """Display the board using cv2 and return a move from the user
    
    Args:
        board (np.ndarray): board state
        moves (set): set of possible moves
        
    Returns:
        tuple: next move
    """
    return cv2_display(board.shape[0], board, moves, adj_cells)

def greedy(board: np.ndarray, moves: set, turn: int) -> tuple:
    pass


def heuristic(board: np.ndarray, turn: int) -> tuple:
    """Which heuristic to use. We have 2 tables"""
    return np.sum(TABLE[np.where(board == 1)]) if turn == 1 else np.sum(TABLE[np.where(board == -1)])

def negamax(board: np.ndarray, adjacent_cells: set, turn: int, depth: int) -> tuple:
    """Return the best move using negamax (minimax where the opponent's score is negated)
    
    Args:
        board (np.ndarray): board state
        moves (set): set of possible moves
        turn (int): current player
        depth (int): depth of the search
        
    Returns:
        tuple: best score, best move
    """
    if depth == MAX_DEPTH or len(adjacent_cells) == 0:
        return [heuristic(board, turn)]
    moves, invalid_directions = get_possible_moves(board, adjacent_cells, turn)
    if len(moves) == 0:
        return [heuristic(board, turn)]
    best = -MAX_INT
    best_move = None
    for move in moves:
        board_copy = board.copy()
        play(board_copy, move, turn, adjacent_cells, invalid_directions)
        score = -negamax(board_copy, adjacent_cells, -turn, depth+1)[0]
        if score == best and random.choice([True, False]):
            best_move = move
        if score > best:
            best = score
            best_move = move
    return best, best_move

# @timit
def get_winner(board: np.ndarray) -> None:
    """Print the winner

    Args:
        board (np.ndarray): board state
    """
    black = np.sum(board == -1)
    white = np.sum(board == 1)
    if black > white:
        print("Black wins" + "(" + str(black) + " vs " + str(white) + ")" )
    elif black < white:
        print("White wins" + "(" + str(white) + " vs " + str(black) + ")" )
    else:
        print("Draw" + "(" + str(black) + " vs " + str(white) + ")" )


def time_n(n:int = 1000, args:list = [0, None, 8, False]):
    """Time the function n times

    Args:
        n (int, optional): number of times to run the function. Defaults to 1000.
    """
    onset = time.time()
    for _ in range(n):
        othello(*args)
    offset = time.time()
    print("Execution time:", offset - onset)
    print("Average time:", (offset - onset)/n)

if __name__ == "__main__":
    # time_n(1000, [0, None, 8, False])
    
    # Bot vs Bot
    start = time.time()
    code, board, moves, adj_cells = othello(0, 1, 8, display=True) 
    end = time.time()
    print("Execution time:", end - start)
    cv2_display(board.shape[0], board, moves, adj_cells, display_only=True, last_display=True)
    # # STATS
    # intermediate_results.to_csv(os.path.join(path_dir, file2_name), index=False)
    # intermediate_results = intermediate_results.groupby('function')
    # # save all intermediate results in 'intermediate_results'
    # for function, group in intermediate_results:
    #     results.loc[len(results)] = [function, group['execution_time'].sum(), group['execution_time'].values]
    # # Sort by execution time
    # results = results.sort_values(by='execution_time', ascending=False)
    # print(results)
    # results.to_csv(os.path.join(path_dir, file_name), index=False)