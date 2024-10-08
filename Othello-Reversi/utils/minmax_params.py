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
class Algorithm:
    # Minimax algorithms
    NONE = 0
    MINIMAX = 1
    ALPHABETA = 2
    NEGAMAX = 3
    NEGAMAX_ALPHA_BETA = 4

    def __init__(self):
        pass


class Strategy:
    # Mode for the player
    HUMAN = 0
    RANDOM = 1
    POSITIONAL = 2
    ABSOLUTE = 3
    MOBILITY = 4
    MIXED = 5

    def __init__(self):
        pass


class Heuristic:
    # Heuristic evaluation functions
    NONE = 0
    TABLE1 = 1
    TABLE2 = 2

    def __init__(self):
        pass
