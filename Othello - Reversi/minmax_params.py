import numpy as np

MAX_DEPTH = 2
#MAX_INT = np.iinfo(np.int32).max
MAX_INT = 10000

TABLE1 = np.array(
    [
        [500, -150, 30, 10, 10, 30, -150, 500],
        [-150, -250, 0, 0, 0, 0, -250, -150],
        [30, 0, 1, 2, 2, 1, 0, 30],
        [10, 0, 2, 16, 16, 2, 0, 10],
        [10, 0, 2, 16, 16, 2, 0, 10],
        [30, 0, 1, 2, 2, 1, 0, 30],
        [-150, -250, 0, 0, 0, 0, -250, -150],
        [500, -150, 30, 10, 10, 30, -150, 500]
    ]
)

TABLE2 = np.array(
    [
        [100, -20, 10, 5, 5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [10, -2, -1, -1, -1, -1, -2, 10],
        [5, -2, -1, -1, -1, -1, -2, 5],
        [5, -2, -1, -1, -1, -1, -2, 5],
        [10, -2, -1, -1, -1, -1, -2, 10],
        [-20, -50, -2, 2, -2, -2, -50, -20],
        [100, -20, 10, 5, 5, 10, -20, 100]
    ]
)
