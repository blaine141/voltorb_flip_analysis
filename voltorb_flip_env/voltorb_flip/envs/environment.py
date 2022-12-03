import gym
from gym import spaces
import random
import numpy as np
from voltorb_flip.solver import Solver


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
    
    return level, np.array(board).reshape(5, 5)

class VoltorbFlipState:
    def __init__(self, level, board, probabilities):
        self.level = level
        self.board = np.copy(board)
        self.probabilities = probabilities

    def numpy(self):
        level_layer = self.level * np.ones((5, 5, 1))
        level_layer[self.board != -1] = -1

        return np.moveaxis(np.concatenate((level_layer, self.probabilities), axis=-1), 2, 0).astype(np.float32)

    def flatten(self):
        return self.numpy().flatten()

    
class VoltorbFlipEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(low=-1, high=8, shape=[5,5,5], dtype=np.float32)
        self.solver: Solver = None
    

    def step(self, action):
        reward = 0
        done = False

        row = action // 5
        column = action % 5

        if self.board[row, column] == 0 or self.visible_board[row, column] != -1:
            reward = -1
            done = True
        else:
            revealed_cell = self.board[row, column]
            self.visible_board[row, column] = revealed_cell
            self.solver.update_board(column, row, revealed_cell)

            if np.sum(self.visible_board == 2) + np.sum(self.visible_board == 3) == np.sum(self.board > 1):
                done=True
                reward = 1

        return self.get_state(), reward, done, {}

    def get_state(self):
        self.solver.compute_probabilities()
        state = VoltorbFlipState(self.level, self.visible_board, self.solver.cell_probabilities)
        return state.numpy()

    def reset(self):
        self.level, self.board = generate_board()
        self.visible_board = -np.ones((5, 5))

        self.row_constraints = []
        for row in range(5):
            self.row_constraints.append([np.sum(self.board[row]), np.sum(self.board[row] == 0)])
        self.column_constraints = []
        for column in range(5):
            self.column_constraints.append([np.sum(self.board[:, column]), np.sum(self.board[:, column] == 0)])

        self.solver = Solver(self.level, self.visible_board, self.row_constraints, self.column_constraints)

        return self.get_state()

    def render(self, mode='human'):
        print(self.visible_board)

    def copy(self):
        env = VoltorbFlipEnv()
        env.level = self.level
        env.board = np.copy(self.board)
        env.visible_board = np.copy(self.visible_board)
        env.row_constraints = self.row_constraints
        env.column_constraints = self.column_constraints
        env.solver = Solver(self.level, self.visible_board, self.row_constraints, self.column_constraints)

        return env


