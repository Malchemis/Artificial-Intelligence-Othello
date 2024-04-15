import yaml
import pickle
import os

from game import othello
from node import replay
from utils.measure import time_n, time_only
from utils.minmax_params import Algorithm, Strategy


def championship(n_iterations=100, stats_path=None):
    """Test multiple configurations using grid search."""
    os.makedirs(stats_path, exist_ok=True)

    default_minimax_algorithm = [Algorithm.NEGAMAX_ALPHA_BETA, Algorithm.NEGAMAX_ALPHA_BETA]
    defaults_threshold = [30, 55]   # Default thresholds for MIXED strategy
    strategies_noh = [Strategy.ABSOLUTE, Strategy.MOBILITY]
    strategies_heu = [Strategy.POSITIONAL, Strategy.MIXED]
    depths = [6]
    tables = [1, 2]  # only consider for POSITIONAL and MIXED strategies
    # Total: 3*(4 + 8 + 8 + 16) = 108 trials
    for depth in depths:
        # Non-heuristic against non-heuristic (2*2 = 4)
        for strategy1 in strategies_noh:
            for strategy2 in strategies_noh:
                _, _, _, _, _ = time_n(
                    othello, n_iterations,
                    ([strategy1, strategy2], default_minimax_algorithm, [depth, depth],
                     [0, 0], defaults_threshold, 8, False, False,
                     stats_path + f"/championship_{strategy1}_{strategy2}_{depth}_{0}_{0}_"
                                  f"{default_minimax_algorithm[0]}.csv")
                )
        # Heuristic against non-heuristic (2*2*2 = 8)
        for strategy1 in strategies_heu:
            for strategy2 in strategies_noh:
                for table in tables:
                    _, _, _, _, _ = time_n(
                        othello, n_iterations,
                        ([strategy1, strategy2], default_minimax_algorithm, [depth, depth],
                         [table, 0], defaults_threshold, 8, False, False,
                         stats_path + f"/championship_{strategy1}_{strategy2}_{depth}_{table}_{0}_"
                                      f"{default_minimax_algorithm[0]}.csv")
                    )
        # Non-heuristic against heuristic (2*2*2 = 8)
        for strategy1 in strategies_noh:
            for strategy2 in strategies_heu:
                for table in tables:
                    _, _, _, _, _ = time_n(
                        othello, n_iterations,
                        ([strategy1, strategy2], default_minimax_algorithm, [depth, depth],
                         [0, table], defaults_threshold, 8, False, False,
                         stats_path + f"/championship_{strategy1}_{strategy2}_{depth}_{0}_{table}_"
                                      f"{default_minimax_algorithm[0]}.csv")
                    )
        # Heuristic against heuristic (2*2*2*2 = 16)
        for strategy1 in strategies_heu:
            for strategy2 in strategies_heu:
                for table1 in tables:
                    for table2 in tables:
                        _, _, _, _, _ = time_n(
                            othello, n_iterations,
                            ([strategy1, strategy2], default_minimax_algorithm, [depth, depth],
                             [table1, table2], defaults_threshold, 8, False, False,
                             stats_path + f"/championship_{strategy1}_{strategy2}_{depth}_{table1}_{table2}_"
                                          f"{default_minimax_algorithm[0]}.csv")
                        )


def complexity_analysis(n_iterations=100, stats_path=None):
    """Test multiple configurations using grid search.
    Measure number of explored nodes."""
    os.makedirs(stats_path, exist_ok=True)

    # We compare only Negamax and Negamax with alpha-beta pruning against random player
    defaults_threshold = [30, 55]   # Default thresholds for MIXED strategy
    default_strategy = Strategy.RANDOM
    default_table = 2  # We only consider the second table (which is the better one)

    depths = [2, 4, 6]
    algorithms = [Algorithm.NEGAMAX, Algorithm.NEGAMAX_ALPHA_BETA]
    strategies = [Strategy.POSITIONAL, Strategy.ABSOLUTE, Strategy.MOBILITY, Strategy.MIXED]
    # Total: 3*2*4 = 24 trials
    for depth in depths:
        for algorithm in algorithms:
            for strategy in strategies:
                _, _, _, _, _ = time_n(
                    othello, n_iterations,
                    ([strategy, default_strategy], [algorithm, 0], [depth, 0],
                     [default_table, 0], defaults_threshold, 8, False, False,
                     stats_path + f"/complexity_{strategy}_{default_strategy}_{depth}_{default_table}_{0}_{algorithm}"
                                  f".csv")
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
    # complexity_analysis(100, "../stats/")
    championship(5, "../stats/")
