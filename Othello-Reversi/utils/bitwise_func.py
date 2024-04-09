def set_state(bitboard: int, x: int, y: int, size: int):
    """Add a bit to the board by shifting a 1 to the left by x * size + y (flattened coordinates)"""
    return bitboard | (1 << (x * size + y))


def cell_count(bitboard: int):
    """Count the number of cells in the board"""
    return bitboard.bit_count()


def bits(number):
    """Generator to get the bits of a number"""
    bit = 1
    while number >= bit:
        if number & bit:
            yield bit
        bit <<= 1


def get_state(bitboard: int, x: int, y: int, size: int):
    """Return the state of the cell by shifting the board to the right by x * size + y and
    taking the least significant bit"""
    return (bitboard >> (x * size + y)) & 1


def get_indexes_move(move: int, size: int):
    """Return i and j indexes of the bitboard for a possible move"""
    position = move.bit_length() - 1
    # Calculate row and column indexes from the position
    i = position // size
    j = position % size
    return i, j


def print_pieces(bitboard: int, size: int):
    """Print the bit values of the board as a matrix of 0 and 1"""
    for i in range(size):
        for j in range(size):
            print(get_state(bitboard, i, j, size), end=' ')
        print()
    print()


def print_board(white: int, black: int, size: int):
    """Print the board with W for white, B for black and . for empty cells"""
    for i in range(size):
        for j in range(size):
            if get_state(white, i, j, size):
                print('W', end=' ')
            elif get_state(black, i, j, size):
                print('B', end=' ')
            else:
                print('.', end=' ')
        print()
    print()
