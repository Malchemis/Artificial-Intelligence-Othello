from next import generate_moves, make_move


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

    def get_child(self, child):
        return self.children[self.children.index(child)]

    def add_other_child(self, other):
        child = Node(self, other.own_pieces, other.enemy_pieces, -self.turn, self.size)
        self.children.append(child)
        return child

    def add_other_child_from_pieces(self, own, enemy):
        child = Node(self, own, enemy, -self.turn, self.size)
        self.children.append(child)
        return child

    def __eq__(self, other):
        return (self.own_pieces == other.own_pieces and self.enemy_pieces == other.enemy_pieces
                and self.turn == other.turn)

    def __repr__(self):
        return f"{self.own_pieces}, {self.enemy_pieces}, {self.turn}"

    def __str__(self):
        return f"{self.own_pieces}, {self.enemy_pieces}, {self.turn}"
