import random
import numpy as np

from visualize import cv2_display

def strategy_bot(strategy_i:int, board: np.ndarray, moves: set, turn: int) -> tuple:
    """Return the next move

    Args:
        strategy_i (int): describe the strategy of the bot. None: random, 0: greedy, 1: minimax
        board (np.ndarray): board state
        moves (set): set of possible moves
        turn (int): current player

    Returns:
        tuple: next move
    """
    if strategy_i is None:
        return random.choice(list(moves))
    elif strategy_i == 0:
        return greedy(board, moves, turn)
    else:
        return minimax(board, moves, turn, strategy_i)

def strategy_human(board: np.ndarray, moves: set, adj_cells=None) -> tuple:
    """Display the board using cv2 and return a move from the user
    
    Args:
        board (np.ndarray): board state
        moves (set): set of possible moves
        
    Returns:
        tuple: next move
    """
    x,y = cv2_display(board.shape[0], board, moves, adj_cells=adj_cells)
    for move in moves:
        if move[0] == x and move[1] == y:
            return move
    return None

def greedy(board: np.ndarray, moves: set, turn: int) -> tuple:
    pass

def minimax(board: np.ndarray, moves: set, turn: int, depth: int) -> tuple:
    pass