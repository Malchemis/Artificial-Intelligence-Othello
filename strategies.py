

# bot_v_bot, bot_v_human, human_v_bot, human_v_human
def bot_v_bot(board: np.ndarray, moves: set, turn: int, strategy_i: int) -> tuple:
    """Return the next move for bot vs bot

    Args:
        board (np.ndarray): board state
        moves (set): set of possible moves
        turn (int): current player
        strategy_i (int): describe the strategy of the bot. None: random, 0: greedy, 1: minimax

    Returns:
        tuple: next move
    """
    if strategy_i is None:
        return random.choice(list(moves))
    if strategy_i == 0:
        return greedy(board, moves, turn)
    if strategy_i == 1:
        return minimax(board, moves, turn)
    return random.choice(list(moves))


def bot_v_human(board: np.ndarray, moves: set, turn: int, strategy_i: int) -> tuple:
    """Return the next move for bot vs human
    Args:
        board (np.ndarray)
        moves (set)
        turn (int)
        strategy_i (int)
    Returns:
        tuple
    """
    if turn == 1:
        if strategy_i is None:
            return random.choice(list(moves))
        if strategy_i == 0:
            return greedy(board, moves, turn)
        if strategy_i == 1:
            return minimax(board, moves, turn)
        return random.choice(list(moves))
    else:
        return None
    return random.choice(list(moves))

