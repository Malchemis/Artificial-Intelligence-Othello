import cProfile
import pstats
import time

from tqdm import tqdm


def profile_n(func, n: int, params: tuple) -> None:
    """Profile the code

    Args:
        func (function): function to profile
        n (int): number of iterations
        params (tuple): parameters of the game
    """
    # code, board, moves, adj_cells = func(*params)
    # profile n calls
    pr = cProfile.Profile()
    pr.enable()
    for _ in tqdm(range(n), desc="Progress", unit="iteration"):
        _, _, _, _ = func(*params)
    pr.disable()
    # sort by cumulative time
    ps = pstats.Stats(pr).sort_stats('cumulative')
    ps.print_stats()


def time_n(func, n: int, params: tuple) -> None:
    """Time the code

    Args:
        func (function): function to time
        n (int): number of iterations
        params (tuple): parameters of the game
    """
    wins = []
    onsets = []
    offsets = []
    for _ in tqdm(range(n), desc="Progress", unit="iteration"):
        onsets.append(time.perf_counter())
        code, own, enemy, turn = func(*params)
        offsets.append(time.perf_counter())
        wins.append(code)

    print("\nTime:", offsets[-1] - onsets[0])
    if n > 1:
        print("Average time:", (offsets[-1] - onsets[0]) / n)
        print("Black won:", wins.count(-1), '(' + str(wins.count(-1) / n * 100) + '%)')
        print("White won:", wins.count(1), '(' + str(wins.count(1) / n * 100) + '%)')
        print("Draw:", wins.count(0), '(' + str(wins.count(0) / n * 100) + '%)')
