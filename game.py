import numpy as np
import random

from visualize import cv2_display
from minmax_params import TABLE, MAX_DEPTH, MAX_INT

import time
import cProfile
import pstats


def othello(mode:int, strategy_i:int=0, size:int = 8, display:bool = False) -> int:
    """Handles the game logic of Othello. We keep track of the board, the turn, the possible moves and the adjacent cells.
    - The game is played on an 8x8 board by default.
    - The game is played by two players, one with the black pieces (value -1) and one with the white pieces (value +1). Empty cells are represented by 0.
    - The game starts with 2 black pieces and 2 white pieces in the center of the board.    

    Args:
        mode (int): describe if it's BotvBot, BotvHuman, HumanvBot or HumanvHuman (0, 1, 2, 3 respectively)
        strategy_i (int, optional): describe the strategy of the bot. Defaults to 0 (~random), 1: minmax, 2: greedy.
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
    while len(adjacent_cells) >= 0:
        time.sleep(1)
        moves, invalid_directions = get_possible_moves(board, adjacent_cells, turn) # set the possible moves
        if len(moves) == 0:
            # verify if the other player can play
            if len(get_possible_moves(board, adjacent_cells, -turn)[0]) == 0:
                break
            turn *= -1
            continue
        copy_turn = turn # we are careful not to modify the turn, specially for minmax
        next_move = strategy(mode, strategy_i, board, moves, copy_turn, adjacent_cells.copy(), display=display)
        play(board, next_move, turn, adjacent_cells, invalid_directions) # flip the cells, update adjacent cells, update possible moves
        turn *= -1
    
    return get_winner(board), board, moves, adjacent_cells


def error_handling(mode:int, strategy_i:int, size:int) -> None:
    """Check if the input parameters are correct

    Args:
        mode (int): describe if it's BotvBot, BotvHuman, HumanvBot or HumanvHuman (0, 1, 2, 3 respectively)
        strategy_i (int): describe the strategy of the bot. 0: random, 1: minmax, 2: greedy
        size (int): size of the board
    """
    if size < 4:
        raise ValueError("Size must be at least 4")
    if size % 2 != 0:
        raise ValueError("Size must be an even number")
    if mode not in [0, 1, 2, 3]:
        raise NotImplementedError("Mode not implemented")
    if strategy_i not in [0, 1, 2]:
        raise NotImplementedError("Strategy not implemented")


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


def strategy(mode:int, strategy_i:int, board: np.ndarray, moves: set, turn: int, adj_cells, display=False) -> tuple:
    """Return the next move

    Args:
        mode (int): describe if it's BotvBot, BotvHuman, HumanvBot or HumanvHuman (0, 1, 2, 3 respectively)
        strategy_i (int): describe the strategy of the bot. 0: random, 1: minmax, 2: greedy
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
    for dx, dy in set([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]) - invalid_directions - {(dx, dy)}:
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
        strategy_i (int): describe the strategy of the bot. 0: random, 1: minmax, 2: greedy
        board (np.ndarray): board state
        moves (set): set of possible moves
        turn (int): current player

    Returns:
        tuple: next move
    """
    if display:
        cv2_display(board.shape[0], board, moves, adj_cells, display_only=True)
    if strategy_i == 0:
        return random.choice(list(moves))
    if strategy_i == 1:
        return negamax(board, adj_cells, turn, strategy_i)[1]
    return greedy(board, adj_cells, turn, strategy_i)[1]

def strategy_human(board: np.ndarray, moves: set, adj_cells) -> tuple:
    """Display the board using cv2 and return a move from the user
    
    Args:
        board (np.ndarray): board state
        moves (set): set of possible moves
        
    Returns:
        tuple: next move
    """
    return cv2_display(board.shape[0], board, moves, adj_cells)

def greedy(board: np.ndarray, adjacent_cells: set, turn: int, depth: int) -> tuple:
    """
    Like minmax, but looks at the number of pieces flipped (tries to maximize (player's pieces - opponent's pieces)) instead of using a heuristic.
    
    Args:
        board (np.ndarray): board state
        adjacent_cells (set): set of adjacent cells
        turn (int): current player
        depth (int): depth of the search
        
    Returns:
        tuple: best score, best move        
    """
    if depth == MAX_DEPTH or len(adjacent_cells) == 0:
        return [np.sum(board == turn) - np.sum(board == -turn)]
    moves, invalid_directions = get_possible_moves(board, adjacent_cells, turn)
    if len(moves) == 0:
        return [np.sum(board == turn) - np.sum(board == -turn)]
    best = -MAX_INT
    best_moves = []
    for move in moves:
        board_copy = board.copy()
        play(board_copy, move, turn, adjacent_cells, invalid_directions)
        score = -greedy(board_copy, adjacent_cells, -turn, depth+1)[0]
        if score > best:
            best = score
            best_moves = [move]
        elif score == best:
            best_moves.append(move)
    best_move = random.choice(best_moves)
    return best, best_move


def heuristic(board: np.ndarray, turn: int) -> tuple:
    """Which heuristic to use. We have 2 tables"""
    return np.sum(TABLE[np.where(board == 1)]) if turn == 1 else np.sum(TABLE[np.where(board == -1)])

def negamax(board: np.ndarray, adjacent_cells: set, turn: int, depth: int) -> tuple:
    """Return the best move using negamax (minmax where the opponent's score is negated)
    
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
    best_moves = []
    for move in moves:
        board_copy = board.copy()
        play(board_copy, move, turn, adjacent_cells, invalid_directions)
        score = -negamax(board_copy, adjacent_cells, -turn, depth+1)[0]
        if score > best:
            best = score
            best_moves = [move]
        elif score == best:
            best_moves.append(move)
    best_move = random.choice(best_moves)
    return best, best_move


def get_winner(board: np.ndarray) -> None:
    """Print the winner

    Args:
        board (np.ndarray): board state
    """
    black = np.sum(board == -1)
    white = np.sum(board == 1)
    if black > white:
        print("Black wins" + "(" + str(black) + " vs " + str(white) + ")" )
        return -1
    if black < white:
        print("White wins" + "(" + str(white) + " vs " + str(black) + ")" )
        return 1
    print("Draw" + "(" + str(black) + " vs " + str(white) + ")" )
    return 0

def profile(params: tuple) -> None:
    """Profile the code

    Args:
        params (tuple): parameters of the game
    """
    prof = cProfile.Profile()
    prof.enable()
    code, board, moves, adj_cells = othello(*params)
    prof.disable()
    stats = pstats.Stats(prof).sort_stats('cumulative')
    stats.print_stats()
    return code, board, moves, adj_cells


def time_n(n: int, params: tuple) -> None:
    """Time the code

    Args:
        n (int): number of iterations
        params (tuple): parameters of the game
    """
    onset = time.time()
    wins = []
    for _ in range(n):
        code, _, _, _ = othello(*params)
        wins.append(code)
    offset = time.time()
    print("Time:", offset-onset)
    if n > 1:
        print("Average time:", (offset-onset)/n)
        print("Black won:", wins.count(-1), '(' + str(wins.count(-1)/n*100) + '%)')
        print("White won:", wins.count(1), '(' + str(wins.count(1)/n*100) + '%)')
        print("Draw:", wins.count(0), '(' + str(wins.count(0)/n*100) + '%)')

if __name__ == "__main__":
    # code, board, moves, adj_cells = profile((0, None, 8, True))
    time_n(1000, (0, 0, 8, True))