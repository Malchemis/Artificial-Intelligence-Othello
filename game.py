import numpy as np
import time
import pandas as pd
import os

from strategies import strategy_bot, strategy_human

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

@timit
def othello(mode:int, strategy_i:int=None, size:int = 8) -> int:
    """Handles the game logic of Othello. We keep track of the board, the turn, the possible moves and the adjacent cells.
    - The game is played on an 8x8 board by default.
    - The game is played by two players, one with the black pieces (value -1) and one with the white pieces (value +1). Empty cells are represented by 0.
    - The game starts with 2 black pieces and 2 white pieces in the center of the board.
    - We keep track of the player's turn 
    

    Args:
        mode (int): describe if it's BotvBot, BotvHuman, HumanvBot or HumanvHuman (0, 1, 2, 3 respectively)
        strategy_i (int, optional): describe the strategy of the bot. Defaults to None (~random), 0: greedy, 1: minimax
        size (int, optional): size of the board. Defaults to 8.
    Returns:
        int: return code. 0 if finished correclty
    """
    # Error Handling
    if size < 4:
        raise ValueError("Size must be at least 4")
    if size % 2 != 0:
        raise ValueError("Size must be an even number")
    if mode not in [0, 1, 2, 3]:
        raise NotImplementedError("Mode not implemented")
    if strategy_i not in [None, 0, 1]:
        raise NotImplementedError("Strategy not implemented")
    
    # init board, turn, adjacent cells, possible moves
    board = np.zeros((size, size), dtype=int)
    adjacent_cells = set()
    turn = -1 # Black starts
    init_board(board)                   # set the starting positions
    init_adjacent_cells(adjacent_cells) # set the adjacent cells
    while game_status(board, adjacent_cells, turn):
        print('--'*20)
        print(f'Turn: {turn}')
        print(f'Adjacent cells: {adjacent_cells}')
        moves = get_possible_moves(board, adjacent_cells, turn) # set the possible moves
        print(f'Possible moves: {moves}')
        next_move = strategy(mode, strategy_i, board, moves, turn, adjacent_cells)
        print(f'Next move: {next_move}')
        play(board, next_move, turn, adjacent_cells) # flip the cells, update adjacent cells, update possible moves
        turn *= -1
    return 0

@timit
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
    
@timit
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

@timit
def game_status(board: np.ndarray, adjacent_cells: set, turn: int) -> bool:
    """Check if the game is finished

    Args:
        board (np.ndarray): board state
        adjacent_cells (set): set of adjacent cells
        turn (int): current player

    Returns:
        bool: True if the game is finished
    """
    if len(get_possible_moves(board, adjacent_cells, turn)) == 0 and len(get_possible_moves(board, adjacent_cells, -turn)) == 0:
        return False
    return True 


@timit
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
    for x, y in adjacent_cells:
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            is_valid, n_jump = is_valid_direction(board, x, y, dx, dy, turn)
            if is_valid:
                possible_moves.add((x, y, n_jump, dx, dy))
                break
    return possible_moves

@timit
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
    while x >= 0 and x < board.shape[0] and y >= 0 and y < board.shape[1] and board[x][y] == -turn:
        x += dx
        y += dy
        n_jump += 1
    return x >= 0 and x < board.shape[0] and y >= 0 and y < board.shape[1] and board[x][y] == turn and n_jump > 0, n_jump

@timit
def strategy(mode:int, strategy_i:int, board: np.ndarray, moves: set, turn: int, adj_cells=None) -> tuple:
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
        return strategy_bot(strategy_i, board, moves, turn)
    else:
        return strategy_human(board, moves, adj_cells=adj_cells)

@timit
def play(board: np.ndarray, move: tuple, turn: int, adjacent_cells: set) -> None:
    """Play the move

    Args:
        board (np.ndarray): board state
        move (tuple): next move
        turn (int): current player
    """
    x, y, n_jump, dx, dy = move
    board[x][y] = turn
    for _ in range(n_jump):
        x += dx
        y += dy
        board[x][y] = turn
    adjacent_cells.discard((x, y))
    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        if x+dx >= 0 and x+dx < board.shape[0] and y+dy >= 0 and y+dy < board.shape[1] and board[x+dx][y+dy] == 0:
            adjacent_cells.update([(x+dx, y+dy)])

if __name__ == "__main__":
    # Human vs Bot
    othello(2)        
    # STATS
    intermediate_results.to_csv(os.path.join(path_dir, file2_name), index=False)
    intermediate_results = intermediate_results.groupby('function')
    # save all intermediate results in 'intermediate_results'
    for function, group in intermediate_results:
        results.loc[len(results)] = [function, group['execution_time'].sum(), group['execution_time'].values]
    # Compute N: the number of iterations for each function, the Mean time, the standard deviation, the minimum and maximum time
    stats = results.groupby('function').agg({'execution_time': ['count', 'mean', 'std', 'min', 'max']})
    stats.columns = ['N', 'Mean', 'Std', 'Min', 'Max']
    print(stats)
    results.to_csv(os.path.join(path_dir, file_name), index=False)