from typing import Any

import numpy as np
from numpy import signedinteger

from next import generate_moves
from bitwise_func import cell_count


def positional(own: int, enemy: int, size: int, table: np.ndarray) -> signedinteger[Any]:
    """Compute the weighted sum of the board using the positional table

    Args:
        own (int): a bit board of the current player
        enemy (int): a bit board of the other player
        size (int): size of the board
        table (np.ndarray): table of values for the heuristic
    """
    # Convert the binary representations to boolean masks
    own_mask = np.array([bool(own & (1 << i)) for i in range(size * size)])
    enemy_mask = np.array([bool(enemy & (1 << i)) for i in range(size * size)])

    # Apply the masks to the table and sum the values
    sum1 = np.sum(table[own_mask])
    sum2 = np.sum(table[enemy_mask])

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
