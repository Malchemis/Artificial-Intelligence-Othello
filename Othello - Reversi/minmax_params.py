import numpy as np

MAX_INT = 100000

TABLE1 = np.array(
    [
        500, -150, 30, 10, 10, 30, -150, 500,
        -150, -250, 0, 0, 0, 0, -250, -150,
        30, 0, 1, 2, 2, 1, 0, 30,
        10, 0, 2, 16, 16, 2, 0, 10,
        10, 0, 2, 16, 16, 2, 0, 10,
        30, 0, 1, 2, 2, 1, 0, 30,
        -150, -250, 0, 0, 0, 0, -250, -150,
        500, -150, 30, 10, 10, 30, -150, 500
    ]
)

TABLE2 = np.array(
    [
        100, -20, 10, 5, 5, 10, -20, 100,
        -20, -50, -2, -2, -2, -2, -50, -20,
        10, -2, -1, -1, -1, -1, -2, 10,
        5, -2, -1, -1, -1, -1, -2, 5,
        5, -2, -1, -1, -1, -1, -2, 5,
        10, -2, -1, -1, -1, -1, -2, 10,
        -20, -50, -2, 2, -2, -2, -50, -20,
        100, -20, 10, 5, 5, 10, -20, 100
    ]
)


# Enums for the strategies
class Strategy:
    # Minimax algorithms
    MINIMAX = 0
    ALPHABETA = 1
    NEGAMAX = 2
    NEGAMAX_ALPHA_BETA = 3

    # Mode for the player
    HUMAN = 0
    RANDOM = 1
    POSITIONAL_TABLE1 = 2
    POSITIONAL_TABLE2 = 3
    ABSOLUTE = 4
    MOBILITY = 5
    MIXED_TABLE1 = 6
    MIXED_TABLE2 = 7

    def __init__(self):
        pass
