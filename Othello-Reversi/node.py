from next import generate_moves, make_move
from utils.visualize import cv2_display


class Node:
    def __init__(self, parent, own_pieces, enemy_pieces, turn, size):
        self.parent = parent
        self.own_pieces = own_pieces
        self.enemy_pieces = enemy_pieces
        self.turn = turn
        self.size = size
        self.children = []
        self.moves = []
        self.directions = {}
        self.value = None
        self.visited = False

    def expand(self):
        self.moves, self.directions = generate_moves(self.own_pieces, self.enemy_pieces, self.size)
        self.visited = True

    def set_child(self, move):
        own, enemy = make_move(self.own_pieces, self.enemy_pieces, move, self.directions)
        child = Node(self, enemy, own, -self.turn, self.size)
        self.children.append(child)
        return child

    def set_children(self):
        for move in self.moves:
            own, enemy = make_move(self.own_pieces, self.enemy_pieces, move, self.directions)
            self.children.append(Node(self, enemy, own, -self.turn, self.size))

    def get_child(self, child):
        return self.children[self.children.index(child)]

    def add_other_child(self, other):
        for child in self.children:
            if child == other:
                return child
        child = Node(self, other.own_pieces, other.enemy_pieces, other.turn, other.size)
        self.children.append(child)
        return child

    def add_other_child_from_pieces(self, own, enemy):
        child = Node(self, own, enemy, -self.turn, self.size)
        self.children.append(child)
        return child

    def invert(self):
        self.enemy_pieces, self.own_pieces, self.turn = self.own_pieces, self.enemy_pieces, -self.turn

    def count_all_children(self):
        count = len(self.children)
        for child in self.children:
            count += child.count_all_children()
        return count

    def __eq__(self, other):
        return (self.own_pieces == other.own_pieces and self.enemy_pieces == other.enemy_pieces
                and self.turn == other.turn)

    def __str__(self):
        return f"{self.own_pieces}, {self.enemy_pieces}, {self.turn}"

    def __hash__(self):
        return hash((self.own_pieces, self.enemy_pieces, self.turn))


def replay(node: Node, size: int, verbose=False) -> list:
    """Replay the game based on the moves of the Node by backtracking the tree and using a LIFO queue"""
    game = []
    count_child = 0
    while node.parent is not None:
        game.append(node)
        node = node.parent
        count_child += len(node.children)
    game.reverse()
    for node in game:
        if not node.moves:
            node.moves = generate_moves(node.own_pieces, node.enemy_pieces, size)[0]
        if verbose:
            print("-------------")
            print(node)
            print(node.moves)
            print(node.value)
            cv2_display(size, node.own_pieces, node.enemy_pieces, node.moves, node.turn, display_only=True)
            print("Press Enter to continue...", end="")
            input()
    return game
