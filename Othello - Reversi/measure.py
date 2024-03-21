import cProfile
import pstats
import time

from tqdm import tqdm
from Node import replay


def print_runs_stats(wins, onsets, offsets, nb_pieces_played, n: int) -> None:
    """Print the stats of the runs

    Args:
        wins (list): list of the wins
        onsets (list): list of the onsets
        offsets (list): list of the offsets
        nb_pieces_played (int): number of pieces played
        n (int): number of iterations
    """
    print("\nTime:", offsets[-1] - onsets[0])
    print("Pieces played:", nb_pieces_played)
    if n > 1:
        print("Average time:", (offsets[-1] - onsets[0]) / n)
        print("Average pieces played:", nb_pieces_played / n)
        print("Black won:", wins.count(-1), '(' + str(wins.count(-1) / n * 100) + '%)')
        print("White won:", wins.count(1), '(' + str(wins.count(1) / n * 100) + '%)')
        print("Draw:", wins.count(0), '(' + str(wins.count(0) / n * 100) + '%)')


def record_run(func, params, onsets, offsets, wins, nodes, nb_pieces_played_sum) -> int:
    """Record a run"""
    onsets.append(time.perf_counter())
    code, own, enemy, nb_pieces_played, node = func(*params)
    offsets.append(time.perf_counter())
    wins.append(code)
    nodes.append(node)
    return nb_pieces_played_sum + nb_pieces_played


def time_n(func: callable, n: int, params: tuple, profile: bool = False) -> tuple:
    """Time the code

    Args:
        func (callable): function to time
        n (int): number of iterations
        params (tuple): parameters of the game
        profile (bool): profile the code with cProfile
    """
    wins = []
    onsets = []
    offsets = []
    nodes = []
    nb_pieces_played_sum = 0

    pr = None
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    for _ in tqdm(range(n), desc="Progress", unit="iteration"):
        nb_pieces_played_sum = record_run(func, params, onsets, offsets, wins, nodes, nb_pieces_played_sum)

    if pr:
        pr.disable()
        ps = pstats.Stats(pr).sort_stats('cumulative')
        ps.print_stats()

    print_runs_stats(wins, onsets, offsets, nb_pieces_played_sum, n)

    return wins, onsets, offsets, nb_pieces_played_sum, nodes


def time_only(func: callable, n: int, params: tuple) -> None:
    """Time the code without recording the stats

    Args:
        func (callable): function to time
        n (int): number of iterations
        params (tuple): parameters of the game
    """
    onset = time.perf_counter()
    for _ in range(n):
        func(*params)
    offset = time.perf_counter()
    print("Executed in :", offset - onset)
