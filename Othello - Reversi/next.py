def generate_moves(own, enemy, size) -> tuple[list, dict]:
    """Generate the possible moves for the current player using bitwise operations"""
    empty = ~(own | enemy)  # Empty squares (not owned by either player)
    unique_moves = []  # List of possible moves
    dir_jump = {}  # Dictionary of moves and the number of pieces that can be captured in each direction

    # Generate moves in all eight directions
    for direction in [north, south, east, west, north_west, north_east, south_west, south_east]:
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


def make_move(own: int, enemy: int, move_to_play: int, directions: dict) -> tuple[int, int]:
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


# ------------------------------------ DIRECTIONS ------------------------------------ #
# Orthogonal directions
def north(x):
    return x >> 8


def south(x):
    return (x & 0x00ffffffffffffff) << 8


def east(x):
    return (x & 0x7f7f7f7f7f7f7f7f) << 1


def west(x):
    return (x & 0xfefefefefefefefe) >> 1


# Diagonal directions
def north_west(x):
    return north(west(x))


def north_east(x):
    return north(east(x))


def south_west(x):
    return south(west(x))


def south_east(x):
    return south(east(x))


opposite_direction = {north: south, south: north, east: west, west: east, north_west: south_east,
                      north_east: south_west,
                      south_west: north_east, south_east: north_west}


def opposite_dir(direction):
    return opposite_direction[direction]
