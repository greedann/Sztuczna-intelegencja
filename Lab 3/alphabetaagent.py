from exceptions import AgentException
import random


class AlphaBetaAgent:
    def __init__(self, my_token='o') -> None:
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        return self.alphabeta(connect4, 4, True)[1]

    def alphabeta(self, connect4, depth, maximizing_player, alpha = float('-inf'), beta = float('inf')):
        if connect4._check_game_over():
            if connect4.wins == None:  # tie
                return 0, None
            elif connect4.wins == self.my_token:  # win
                return 1, None
            elif connect4.wins != self.my_token:  # lose
                return -1, None

        if depth == 0:
            return connect4.assess(self.my_token), None
        
        if maximizing_player:
            v = float('-inf')
            best_move = None
            available_moves = connect4.possible_drops()
            random.shuffle(available_moves)
            for move in available_moves:
                connect4_copy = connect4.copy()
                connect4_copy.drop_token(move)
                score = self.alphabeta(connect4_copy, depth-1, False, alpha, beta)[0]
                if score > v:
                    v = score
                    best_move = move
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return v, best_move
        else:
            v = float('inf')
            best_move = None
            available_moves = connect4.possible_drops()
            random.shuffle(available_moves)
            for move in available_moves:
                connect4_copy = connect4.copy()
                connect4_copy.drop_token(move)
                score = self.alphabeta(connect4_copy, depth-1, True, alpha, beta)[0]
                if score < v:
                    v = score
                    best_move = move
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return v, best_move

