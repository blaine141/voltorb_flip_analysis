import gym
from gym import spaces
import random
import numpy as np


LEVEL_DATA = [
    [1, 3, 1, 6],
    [1, 0, 3, 6],
    [1, 5, 0, 6],
    [1, 2, 2, 6],
    [1, 4, 1, 6],
    [2, 1, 3, 7],
    [2, 6, 0, 7],
    [2, 3, 2, 7],
    [2, 0, 4, 7],
    [2, 5, 1, 7],
    [3, 2, 3, 8],
    [3, 7, 0, 8],
    [3, 4, 2, 8],
    [3, 1, 4, 8],
    [3, 6, 1, 8],
    [4, 3, 3, 8],
    [4, 0, 5, 8],
    [4, 8, 0, 10],
    [4, 5, 2, 10],
    [4, 2, 4, 10],
    [5, 7, 1, 10],
    [5, 4, 3, 10],
    [5, 1, 5, 10],
    [5, 9, 0, 10],
    [5, 6, 2, 10],
    [6, 3, 4, 10],
    [6, 0, 6, 10],
    [6, 8, 1, 10],
    [6, 5, 3, 10],
    [6, 2, 5, 10],
    [7, 7, 2, 10],
    [7, 4, 4, 10],
    [7, 1, 6, 13],
    [7, 9, 1, 13],
    [7, 6, 3, 10],
    [8, 0, 7, 10],
    [8, 8, 2, 10],
    [8, 5, 4, 10],
    [8, 2, 6, 10],
    [8, 7, 3, 10]
]

def generate_board(level=None):
    levels = np.copy(LEVEL_DATA)
    if level is not None:
        levels = levels[levels[:, 0] == level]

    level, twos, threes, bombs = random.choice(levels)
    board = []
    board += [2] * twos
    board += [3] * threes
    board += [0] * bombs
    board += [1] * (25 - len(board))
    random.shuffle(board)
    board = [2, 2, 0, 0, 0, 2, 2, 0, 2, 0, 3, 0, 1, 2, 0, 0, 0, 0, 1, 2, 0, 0, 2, 0, 2]

    return level, np.array(board).reshape(5, 5)

class VoltorbFlipState:
    def __init__(self, level, board, row_constraints, col_constraints):
        self.level = level
        self.board = board
        self.row_constraints = np.array(row_constraints)
        self.col_constraints = np.array(col_constraints)

    def flatten(self):

        return np.concatenate((self.board.flatten(), self.row_constraints.flatten(), self.col_constraints.flatten(), [self.level]))

        board = self.board.reshape((5,5,1))
        row_constraints = np.stack([self.row_constraints] * 5, axis=1)
        col_constraints = np.stack([self.col_constraints] * 5, axis=0)
        level = np.ones((5, 5, 1)) * self.level

        return np.moveaxis(np.concatenate([board, col_constraints, row_constraints, level], axis=2), 2, 0)

    
class VoltorbFlipEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(26)
    

    def step(self, action):
        reward = 0
        done = False

        # If the action is to quit
        if action == 25:
            done = self.end_level()
        else:
            row = action // 5
            column = action % 5

            if self.board[row, column] == 0:
                # reward = -1
                reward = -self.score
                done = True#self.end_level()
            else:
                revealed_cell = self.board[row, column]
                self.visible_board[row, column] = revealed_cell
                self.flipped_tiles += 1
                

                # if revealed_cell == 1:
                #     reward = 0.1
                # elif revealed_cell > 1:
                #     reward = 0.3

                reward = self.score * (revealed_cell - 1)
                self.score += reward

                if np.sum(self.visible_board == 2) + np.sum(self.visible_board == 3) == np.sum(self.board > 1):
                    done = self.end_level()


        return self.get_state(), reward, done

    def end_level(self):
        if self.flipped_tiles < self.level:
            done = True
        else:
            done = False
            self.reset(self.level)
        return done

    def get_state(self):
        state = VoltorbFlipState(self.level, self.visible_board, self.row_constraints, self.column_constraints)
        return state

    def reset(self, level=None, mid_game=False):
        self.level, self.board = generate_board(level)
        self.score = 1
        self.flipped_tiles = 0
        self.visible_board = -np.ones((5, 5))

        self.row_constraints = []
        for row in range(5):
            self.row_constraints.append([np.sum(self.board[row]), np.sum(self.board[row] == 0)])
        self.column_constraints = []
        for column in range(5):
            self.column_constraints.append([np.sum(self.board[:, column]), np.sum(self.board[:, column] == 0)])

        if mid_game:
            tiles_to_flip = random.randint(0, np.sum(self.board >= 1))
            for _ in range(tiles_to_flip):
                row = random.randint(0, 4)
                column = random.randint(0, 4)
                if self.board[row, column] >= 1:
                    self.visible_board[row, column] = self.board[row, column]
                    self.flipped_tiles += 1

        return self.get_state()
