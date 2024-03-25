from exceptions import AgentException
import random


class MinMaxAgent:
    def __init__(self, my_token='o') -> None:
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return self.minimax(connect4, 4, True)[1]

    def minimax(self, connect4, depth, maximizing_player):
        if connect4._check_game_over():
            if connect4.wins == None:  # tie
                return 0, None
            elif connect4.wins == self.my_token:  # win
                return 1, None
            elif connect4.wins != self.my_token:  # lose
                return -1, None

        if depth == 0:
            return connect4.assess(self.my_token), None

        best_score = float('-inf') if maximizing_player else float('inf')
        best_move = None

        available_moves = connect4.possible_drops()
        random.shuffle(available_moves)
        for move in available_moves:
            connect4_copy = connect4.copy()
            connect4_copy.drop_token(move)
            score = self.minimax(connect4_copy, depth-1,
                                 not maximizing_player)[0]
            if maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
        return best_score, best_move
