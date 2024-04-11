import yaml
import pickle
import os

from game import othello
from node import replay
from utils.measure import time_n, time_only
from utils.minmax_params import Strategy


def no_heuristics_strategy(n_iterations=100, stats_path=None):
    """Test multiple configurations using grid search.
    The objective is to analyse performance relative to algorithms and strategies that don't need any heuristics."""
    os.makedirs(stats_path, exist_ok=True)

    default_minimax_strategy = [Strategy.NEGAMAX_ALPHA_BETA, Strategy.NEGAMAX_ALPHA_BETA]
    defaults_threshold = [30, 55]
    algorithms = [Strategy.RANDOM, Strategy.ABSOLUTE, Strategy.MOBILITY]
    depths = [2, 4, 6]
    for depth1 in depths:
        for depth2 in depths:
            for algorithm1 in algorithms:
                for algorithm2 in algorithms:
                    _, _, _, _, _ = time_n(
                        othello, n_iterations,
                        ([algorithm1, algorithm2], default_minimax_strategy, [depth1, depth2],
                         [0, 0], defaults_threshold, 8, False, False,
                         stats_path + f"/no_heuristics_table_{algorithm1}_{algorithm2}_{depth1}_{depth2}.csv")
                    )


def default_run():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    params = (
        config["mode"],
        config["minimax_mode"],
        config["max_depth"],
        config["h_table"],
        config["threshold"],
        config["size"],
        config["display"],
        config["verbose"],
        config["stats_path"]
    )

    os.makedirs(config["stats_path"], exist_ok=True)

    if config["time_only"]:
        time_only(othello, config["n"], params)  # Only time the function
        return

    # Time, register results, and display some information depending on the verbose level.
    wins, onsets, offsets, total_pieces, nodes = time_n(othello, config["n"], params, config["profile"])

    if config["replay"]:  # Only replay the last game
        replay(nodes[-1], config["size"], config[replay] == 2)

    if config["save"]:
        for node in nodes:
            list_of_states = replay(node, config["size"], False)
            os.makedirs(config["save"], exist_ok=True)
            with open(os.path.join(config["save"], f"{hash(node)}.pkl"), "wb") as file:
                pickle.dump(list_of_states, file)


if __name__ == "__main__":
    # default_run()
    no_heuristics_strategy(100, "../stats/")
