from typing import Any

import numpy as np
from numpy import signedinteger

from utils.bitwise_func import cell_count
from next import generate_moves


def positional(own: int, enemy: int, size: int, table: np.ndarray) -> signedinteger[Any]:
    """Compute the weighted sum of the board using the positional table

    Args:
        own (int): a bit board of the current player
        enemy (int): a bit board of the other player
        size (int): size of the board
        table (np.ndarray): table of values for the heuristic
    """
    sum1 = 0
    sum2 = 0
    for i in range(size * size):
        if own & (1 << i):
            sum1 += table[i]
        elif enemy & (1 << i):
            sum2 += table[i]
    return sum1 - sum2


def absolute(own: int, enemy: int, size=None, table=None) -> signedinteger[Any]:
    """Compute the difference between the number of pieces of the current player and the other player

    Args:
        own (int): a bit board of the current player
        enemy (int): a bit board of the other player
        size (int): not used here. It is only to match the signature of the other heuristics
        table (np.ndarray): not used here. It is only to match the signature of the other heuristics
    """
    return cell_count(own) - cell_count(enemy)


def mobility(own: int, enemy: int, size: int, table=None) -> signedinteger[Any]:
    """Compute the difference between the number of possible moves of the current player and the other player

    Args:
        own (int): a bit board of the current player
        enemy (int): a bit board of the other player
        size (int): size of the board
        table (np.ndarray): not used here. It is only to match the signature of the other heuristics
    """
    own_moves, _ = generate_moves(own, enemy, size)
    enemy_moves, _ = generate_moves(enemy, own, size)
    return len(own_moves) - len(enemy_moves)
