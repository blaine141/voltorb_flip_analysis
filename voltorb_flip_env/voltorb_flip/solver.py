from random import random, choice
from typing import List
import numpy as np
from fast_board_generator import find_valid_bombs, find_valid_boards

EPS = 1e-8

TOTAL_BOMB_MAPS = 400
TOTAL_VALID_BOARDS = 200

CELL_DISTRIBUTION = {
    (0, 0): [0, 0, 0],
    (1, 1): [1.000000, 0.000000, 0.000000],
    (1, 2): [0.000000, 1.000000, 0.000000],
    (1, 3): [0.000000, 0.000000, 1.000000],
    (2, 2): [1.000000, 0.000000, 0.000000],
    (2, 3): [0.500000, 0.500000, 0.000000],
    (2, 4): [0.332850, 0.334300, 0.332850],
    (2, 5): [0.000000, 0.500000, 0.500000],
    (2, 6): [0.000000, 0.000000, 1.000000],
    (3, 3): [1.000000, 0.000000, 0.000000],
    (3, 4): [0.666667, 0.333333, 0.000000],
    (3, 5): [0.497900, 0.337533, 0.164567],
    (3, 6): [0.286533, 0.426933, 0.286533],
    (3, 7): [0.165333, 0.336000, 0.498667],
    (3, 8): [0.000000, 0.333333, 0.666667],
    (3, 9): [0.000000, 0.000000, 1.000000],
    (4, 4): [1.000000, 0.000000, 0.000000],
    (4, 5): [0.750000, 0.250000, 0.000000],
    (4, 6): [0.601075, 0.297850, 0.101075],
    (4, 7): [0.438250, 0.373500, 0.188250],
    (4, 8): [0.313925, 0.372150, 0.313925],
    (4, 9): [0.186425, 0.377150, 0.436425],
    (4, 10): [0.098500, 0.303000, 0.598500],
    (4, 11): [0.000000, 0.250000, 0.750000],
    (4, 12): [0.000000, 0.000000, 1.000000],
    (5, 5): [1.000000, 0.000000, 0.000000],
    (5, 6): [0.800000, 0.200000, 0.000000],
    (5, 7): [0.666840, 0.266320, 0.066840],
    (5, 8): [0.534680, 0.330640, 0.134680],
    (5, 9): [0.421440, 0.357120, 0.221440],
    (5, 10): [0.313280, 0.373440, 0.313280],
    (5, 11): [0.222960, 0.354080, 0.422960],
    (5, 12): [0.132220, 0.335560, 0.532220],
    (5, 13): [0.065400, 0.269200, 0.665400],
    (5, 14): [0.000000, 0.200000, 0.800000],
    (5, 15): [0.000000, 0.000000, 1.000000],
}

BOMB_DISTRIBUTION = {
    (1, 0): [1.000000, 0.000000],
    (1, 1): [0.000000, 1.000000],
    (2, 0): [1.000000, 0.000000],
    (2, 1): [0.500000, 0.500000],
    (2, 2): [0.000000, 1.000000],
    (3, 0): [1.000000, 0.000000],
    (3, 1): [0.666667, 0.333333],
    (3, 2): [0.333333, 0.666667],
    (3, 3): [0.000000, 1.000000],
    (4, 0): [1.000000, 0.000000],
    (4, 1): [0.750000, 0.250000],
    (4, 2): [0.500000, 0.500000],
    (4, 3): [0.250000, 0.750000],
    (4, 4): [0.000000, 1.000000],
    (5, 0): [1.000000, 0.000000],
    (5, 1): [0.800000, 0.200000],
    (5, 2): [0.600000, 0.400000],
    (5, 3): [0.400000, 0.600000],
    (5, 4): [0.200000, 0.800000],
    (5, 5): [0.000000, 1.000000],
}

class Constr:
    def __init__(self, total, bomb_count) -> None:
        self.total = total
        self.bomb_count = bomb_count
        self.cell_indicies = np.array([])

    def check_total(self, board) -> bool:
        cells = board[self.cell_indicies]
        return np.sum(cells) == self.total

    def check_bombs(self, board) -> bool:
        cells = board[self.cell_indicies]
        return np.count_nonzero(cells) == 5 - self.bomb_count



class Move:
    def __init__(self, move, cell_probabilities) -> None:
        self.move = move
        self.cell_probabilities = cell_probabilities
        


class Solver:

    def __init__(self, level, initial_board, row_constraints, col_constraints):
        self.level = level
        self.board = np.array(initial_board, int)
        self.row_total_constraints = np.ascontiguousarray(np.array(row_constraints, int)[:,0])
        self.row_bomb_constraints = np.ascontiguousarray(np.array(row_constraints, int)[:,1])
        self.col_total_constraints = np.ascontiguousarray(np.array(col_constraints, int)[:,0])
        self.col_bomb_constraints = np.ascontiguousarray(np.array(col_constraints, int)[:,1])
        self.row_constraints = [Constr(x[0], x[1]) for x in row_constraints]
        self.col_constraints = [Constr(x[0], x[1]) for x in col_constraints]
        self.possible_bomb_maps = []
        self.possible_boards = []

        # Initialize constraints
        self.constraints: List[Constr] = []
        for i in range(5):
            self.row_constraints[i].cell_indicies = (np.array([i, i, i, i, i]), np.array(range(5)))
            self.col_constraints[i].cell_indicies = (np.array(range(5)), np.array([i, i, i, i, i]))
            self.constraints.append(self.row_constraints[i])
            self.constraints.append(self.col_constraints[i])

        
    def update_board(self, x, y, value):
        self.board[y, x] = value
        self.possible_bomb_maps = list(filter(lambda arr: arr[y,x] == (value == 0), self.possible_bomb_maps))
        self.possible_boards = list(filter(lambda arr: arr[y,x] == value, self.possible_boards))

    def compute_probabilities(self):
        move_made = True

        while move_made:
            move_made = False
            
            num_probabilities = np.ones((5,5,3), float)
            bomb_probabilities = np.ones((5,5,2), float)

            # Update number probabilities
            for constraint in self.constraints:
                bombs_left = constraint.bomb_count
                desired_total = constraint.total
                unknown_cells = 5

                applicable_cells = self.board[constraint.cell_indicies]
                unknown_cells -= np.sum(applicable_cells >= 0)
                bombs_left -= np.sum(applicable_cells == 0)
                desired_total -= np.sum(applicable_cells[applicable_cells >= 0])

                if unknown_cells == 0:
                    continue

                bomb_prob = BOMB_DISTRIBUTION[unknown_cells, bombs_left]
                for y,x in zip(*constraint.cell_indicies):
                    if self.board[y,x] == -1:
                        bomb_probabilities[y,x] = np.multiply(bomb_probabilities[y,x], bomb_prob)

                cell_probabilities = CELL_DISTRIBUTION[unknown_cells - bombs_left, desired_total]
                for y,x in zip(*constraint.cell_indicies):
                    if self.board[y,x] == -1:
                        num_probabilities[y,x] = np.multiply(num_probabilities[y,x], cell_probabilities)


            for y in range(5):
                for x in range(5):
                    row_constraints = self.row_constraints[y]
                    col_constraints = self.col_constraints[x]
                    if sum(bomb_probabilities[y,x]) != 0:
                        bomb_probabilities[y,x] /= sum(bomb_probabilities[y,x])
                    if sum(num_probabilities[y,x]) != 0:
                        num_probabilities[y,x] /= sum(num_probabilities[y,x])
                    
                    if bomb_probabilities[y,x,1] == 1:
                        self.update_board(x, y, 0)
                        move_made = True

                    if sum(num_probabilities[y,x]) == 0:
                        # This is a cell that we discovered cant be a number because row and column disagree.
                        self.update_board(x, y, 0)
                        move_made = True

                    
                    if bomb_probabilities[y,x,1] == 0:
                        if num_probabilities[y,x,0] == 1:
                            self.update_board(x, y, 1)
                            move_made = True
                        elif num_probabilities[y,x,1] == 1:
                            self.update_board(x, y, 2)
                            move_made = True
                        elif num_probabilities[y,x,2] == 1:
                            self.update_board(x, y, 3)
                            move_made = True

                    

                    if move_made:
                        break
                if move_made:
                    break


        bomb_layout = np.zeros((5, 5), dtype=int)
        bomb_layout[self.board == -1] = -1
        bomb_layout[self.board == 0] = 1
        bomb_layout[self.board > 0] = 0

        bomb_probabilities = np.ascontiguousarray( bomb_probabilities[:,:,1])

        num_bomb_maps = TOTAL_BOMB_MAPS - len(self.possible_bomb_maps)
        new_bomb_maps = find_valid_bombs(bomb_probabilities, bomb_layout, self.row_bomb_constraints, self.col_bomb_constraints, num_bomb_maps)
        self.possible_bomb_maps += list(new_bomb_maps)

        num_boards = TOTAL_VALID_BOARDS - len(self.possible_boards)
        new_boards = find_valid_boards(num_probabilities, np.array(self.possible_bomb_maps), self.board, self.row_total_constraints, self.col_total_constraints, num_boards)
        self.possible_boards += list(new_boards)
        
        self.cell_probabilities = np.zeros((5,5,4))
        for i in range(4):
            self.cell_probabilities[:,:,i] = np.mean(np.array(self.possible_boards) == i, axis=0)

    
    def reset(self):
        self.possible_bomb_maps.clear()
        self.possible_boards.clear()
        self.compute_probabilities()

    def determine_best_move(self):

        unsolved_cells = self.board == -1
        
        entropy = -np.sum(np.multiply(self.cell_probabilities[:,:,1:], np.log2(self.cell_probabilities[:,:,1:]+EPS)), axis=2)
        selection_score = (entropy+0.001)/(self.cell_probabilities[:,:,0]+EPS) * unsolved_cells

        move = np.unravel_index(np.argmax(selection_score, keepdims=True), selection_score.shape)

        return Move(move, self.cell_probabilities[move][0,0])

if __name__ == "__main__":

    import os
    os.system('cls')
        
    X = -1

    """
        [X, X, X, X, X],
        [X, X, X, X, X],
        [X, X, X, X, X],
        [X, X, X, X, X],
        [X, X, X, X, X]
    """

    initial_board = np.array([
        [X, X, X, X, X],
        [X, X, X, X, X],
        [X, X, X, X, X],
        [X, X, X, X, X],
        [X, X, X, X, X]
    ], dtype=object)

    row_constraints = [
        Constr(5, 2),
        Constr(5, 1),
        Constr(5, 1),
        Constr(4, 3),
        Constr(4, 3)
    ]

    col_constraints = [
        Constr(5,2), Constr(2,4), Constr(6,1), Constr(7,1), Constr(3,2)
    ]


        # np.savetxt("prob_scoring.csv", prob_scoring, delimiter=',')
        # np.savetxt("prob_blowup.csv", prob_blowup, delimiter=',')
        # np.savetxt("entropy.csv", entropy, delimiter=',')