import yaml

from game import othello
from utils.measure import time_n, time_only
from node import replay


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    if config["time_only"]:
        time_only(othello, config["n"], (config["minimax_mode"], config["mode"], config["size"], config["max_depth"],
                                         config["display"], config["verbose"]))
        return

    wins, onsets, offsets, nb_pieces_played_sum, nodes = time_n(
        othello,
        config["n"],
        (config["minimax_mode"], config["mode"], config["size"], config["max_depth"], config["display"],
         config["verbose"]),
        profile=config["profile"]
    )

    if config["replay"]:
        replay(nodes[0], config["size"])


if __name__ == "__main__":
    main()
