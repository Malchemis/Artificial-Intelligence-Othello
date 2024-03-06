from typing import Tuple


def generate_moves(own, enemy, size) -> Tuple[list, dict]:
    """Generate the possible moves for the current player using bitwise operations"""
    empty = ~(own | enemy)  # Empty squares (not owned by either player)
    unique_moves = []  # List of possible moves
    dir_jump = {}  # Dictionary of moves and the number of pieces that can be captured in each direction

    # Generate moves in all eight directions
    for direction in [N, S, E, W, NW, NE, SW, SE]:
        # We get the pieces that are next to an enemy piece in the direction
        count = 0
        victims = direction(own) & enemy
        if not victims:
            continue

        # We keep getting the pieces that are next to an enemy piece in the direction
        for _ in range(size):
            count += 1
            next_piece = direction(victims) & enemy
            if not next_piece:
                break
            victims |= next_piece

        # We get the pieces that can be captured in the direction
        captures = direction(victims) & empty
        # if there are multiple pieces in captures, we separate them and add them to the set
        while captures:
            capture = captures & -captures  # get the least significant bit
            captures ^= capture  # remove the lsb
            if capture not in dir_jump:
                unique_moves.append(capture)
                dir_jump[capture] = []
            dir_jump[capture].append((direction, count))

    return unique_moves, dir_jump


def make_move(own, enemy, move_to_play, directions, size):
    """Make the move and update the board using bitwise operations."""
    for direction, count in directions[move_to_play]:
        victims = move_to_play  # Init the victims with the move to play

        op_dir = opposite_dir(direction)  # opposite direction since we go from the move to play to the captured pieces
        for _ in range(count):
            victims |= (op_dir(victims) & enemy)
        own ^= victims
        enemy ^= victims & ~move_to_play
    # because of the XOR, the move to play which is considered a victim can be returned a pair number of times
    own |= move_to_play
    return own, enemy


def N(x):
    return (x & 0x00ffffffffffffff) << 8


def S(x):
    return (x & 0xffffffffffffff00) >> 8


def E(x):
    return (x & 0x7f7f7f7f7f7f7f7f) << 1


def W(x):
    return (x & 0xfefefefefefefefe) >> 1


def NW(x):
    return N(W(x))


def NE(x):
    return N(E(x))


def SW(x):
    return S(W(x))


def SE(x):
    return S(E(x))


def opposite_dir(direction):
    if direction == N:
        return S
    if direction == S:
        return N
    if direction == E:
        return W
    if direction == W:
        return E
    if direction == NW:
        return SE
    if direction == NE:
        return SW
    if direction == SW:
        return NE
    if direction == SE:
        return NW
