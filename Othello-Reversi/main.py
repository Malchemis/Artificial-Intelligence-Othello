import yaml
import pickle
import os

from game import othello
from node import replay
from utils.measure import time_n, time_only
from utils.minmax_params import Strategy


def experiment_run():
    """Test multiple configurations using grid search.
    The objective is to analyse performance relative to algorithms, depth, heuristics, etc."""
    n_iterations = 10
    default_minimax_strategy = Strategy.NEGAMAX_ALPHA_BETA
    algorithms = [Strategy.RANDOM, Strategy.POSITIONAL, Strategy.ABSOLUTE, Strategy.MOBILITY, Strategy.MIXED]
    depths = [2, 4, 6]
    h_tables = [1, 2]  # table 1 or 2, only for POSITIONAL and MIXED


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

    if config["time_only"]:
        time_only(othello, config["n"], params) # Only time the function
        return

    # Time, register results, and display some information depending on the verbose level.
    wins, onsets, offsets, total_pieces, nodes = time_n(othello, config["n"], params, config["profile"])

    if config["replay"]: # Only replay the last game
        replay(nodes[-1], config["size"], config[replay] == 2)
        
    if config["save"]:
        for node in nodes:
            list_of_states = replay(node, config["size"], False)
            os.makedirs(config["save"], exist_ok=True)
            with open(os.path.join(config["save"], f"{hash(node)}.pkl"), "wb") as file:
                pickle.dump(list_of_states, file)


if __name__ == "__main__":
    default_run()
