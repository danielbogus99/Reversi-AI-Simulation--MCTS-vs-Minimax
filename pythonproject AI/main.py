import random
import math
import copy

SIZE = 8

directions = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1)
]

def initialize_board():
    board = [['.' for _ in range(SIZE)] for _ in range(SIZE)]
    board[SIZE // 2 - 1][SIZE // 2 - 1] = 'W'
    board[SIZE // 2][SIZE // 2] = 'W'
    board[SIZE // 2 - 1][SIZE // 2] = 'B'
    board[SIZE // 2][SIZE // 2 - 1] = 'B'
    return board

def print_board(board):
    for row in board:
        print(' '.join(row))
    print()

def is_valid_move(board, x, y, player):
    if board[x][y] != '.':
        return False
    opponent = 'B' if player == 'W' else 'W'
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        has_opponent_between = False
        while 0 <= nx < SIZE and 0 <= ny < SIZE and board[nx][ny] == opponent:
            nx += dx
            ny += dy
            has_opponent_between = True
        if has_opponent_between and 0 <= nx < SIZE and 0 <= ny < SIZE and board[nx][ny] == player:
            return True
    return False

def get_valid_moves(board, player):
    return [(x, y) for x in range(SIZE) for y in range(SIZE) if is_valid_move(board, x, y, player)]

def make_move(board, x, y, player):
    board[x][y] = player
    opponent = 'B' if player == 'W' else 'W'
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        discs_to_flip = []
        while 0 <= nx < SIZE and 0 <= ny < SIZE and board[nx][ny] == opponent:
            discs_to_flip.append((nx, ny))
            nx += dx
            ny += dy
        if 0 <= nx < SIZE and 0 <= ny < SIZE and board[nx][ny] == player:
            for fx, fy in discs_to_flip:
                board[fx][fy] = player

class MCTSNode:
    def __init__(self, board, player, parent=None, move=None):
        self.board = board
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(get_valid_moves(self.board, self.player))

    def is_terminal(self):
        return not get_valid_moves(self.board, self.player)

    def best_child(self, exploration_weight=1.41):
        if not self.children:
            return None
        choices_weights = [
            (child.value / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children if child.visits > 0
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        valid_moves = get_valid_moves(self.board, self.player)
        cPW = 1.5
        if len(self.children) < int(self.visits ** (1 / cPW)) and len(self.children) < len(valid_moves):
            move = valid_moves[len(self.children)]
            new_board = copy.deepcopy(self.board)
            make_move(new_board, move[0], move[1], self.player)
            child_node = MCTSNode(new_board, 'B' if self.player == 'W' else 'W', self, move)
            self.children.append(child_node)
            return child_node
        return None

    def rollout(self):
        current_board = copy.deepcopy(self.board)
        current_player = self.player
        while get_valid_moves(current_board, current_player):
            valid_moves = get_valid_moves(current_board, current_player)
            best_move = max(valid_moves, key=lambda move: sum(row.count(current_player) for row in current_board))
            make_move(current_board, best_move[0], best_move[1], current_player)
            current_player = 'B' if current_player == 'W' else 'W'
        black_score = sum(row.count('B') for row in current_board)
        white_score = sum(row.count('W') for row in current_board)
        return 1 if black_score > white_score else 0 if white_score > black_score else 0.5

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

    def best_action(self):
        simulation_depth = 5
        for _ in range(simulation_depth):
            leaf = self.tree_policy()
            if leaf:
                result = leaf.rollout()
                leaf.backpropagate(result)
        return self.best_child(exploration_weight=0).move if self.children else None

    def tree_policy(self):
        current_node = self
        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                return current_node.expand() or current_node
            else:
                current_node = current_node.best_child()
        return current_node

class Minimax:
    def __init__(self, depth):
        self.depth = depth

    def evaluate_board(self, board, player):
        return sum(row.count(player) for row in board)

    def order_moves(self, board, moves, player):
        return sorted(moves, key=lambda move: self.evaluate_board(copy.deepcopy(board), player), reverse=True)

    def minimax(self, board, depth, maximizing_player, player, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or not get_valid_moves(board, player):
            return self.evaluate_board(board, player)
        opponent = 'B' if player == 'W' else 'W'
        valid_moves = get_valid_moves(board, player)
        ordered_moves = self.order_moves(board, valid_moves, player)
        if maximizing_player:
            max_eval = float('-inf')
            for move in ordered_moves:
                new_board = copy.deepcopy(board)
                make_move(new_board, move[0], move[1], player)
                max_eval = max(max_eval, self.minimax(new_board, depth - 1, False, opponent, alpha, beta))
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                new_board = copy.deepcopy(board)
                make_move(new_board, move[0], move[1], player)
                min_eval = min(min_eval, self.minimax(new_board, depth - 1, True, opponent, alpha, beta))
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            return min_eval


# Game simulation
num_games = int(input("Enter the number of games to play: "))
mcts_player = input("Choose the player for MCTS (B or W): ").upper()
minimax_player = 'B' if mcts_player == 'W' else 'W'
mcts_wins = 0
minimax_wins = 0

minimax_ai = Minimax(depth=3)  # Increased depth for better performance

for game in range(num_games):
    print(f"\nStarting Game {game + 1}...")
    board = initialize_board()
    current_player = 'B'

    while get_valid_moves(board, 'B') or get_valid_moves(board, 'W'):
        print_board(board)
        valid_moves = get_valid_moves(board, current_player)

        if not valid_moves:  # Skip turn if no valid moves
            print(f"No valid moves for {current_player}. Skipping turn.")
            current_player = 'B' if current_player == 'W' else 'W'
            continue

        if current_player == mcts_player:  # MCTS
            print("MCTS is thinking...")
            root = MCTSNode(board, mcts_player)
            move = root.best_action()
        else:  # Minimax
            print("Minimax is thinking...")
            best_score = float('-inf')
            best_move = None
            for move in valid_moves:
                new_board = copy.deepcopy(board)
                make_move(new_board, move[0], move[1], minimax_player)
                score = minimax_ai.minimax(new_board, minimax_ai.depth, False, mcts_player)
                if score > best_score:
                    best_score = score
                    best_move = move
            move = best_move

        if move is not None:  # Perform the move
            make_move(board, move[0], move[1], current_player)

        current_player = 'B' if current_player == 'W' else 'W'

    # Final result
    print_board(board)
    black_score = sum(row.count('B') for row in board)
    white_score = sum(row.count('W') for row in board)
    print(f"Final Score - Black: {black_score}, White: {white_score}")

    if black_score > white_score:
        if mcts_player == 'B':
            mcts_wins += 1
        else:
            minimax_wins += 1
    elif white_score > black_score:
        if mcts_player == 'W':
            mcts_wins += 1
        else:
            minimax_wins += 1

print(f"\nResults after {num_games} games:")
print(f"MCTS Wins: {mcts_wins}")
print(f"Minimax Wins: {minimax_wins}")