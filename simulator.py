import numpy as np
from csv import writer
import gym
from RL.vanilla_policy_gradient_model import VanillaPolicyGradient
from glob import glob
import os
import re
from tqdm import tqdm
import voltorb_flip

MOVE_LOG_FILE = "moves.csv"
GAME_LOG_FILE = "game.csv"
DEVIATION_LOG_FILE = "deviation.csv"
EPS = 1e-8
NUM_GAMES = 20_001

def append_to_csv(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def greedy_policy(state):
    unsolved_cells = state[0] != -1
    prob_scoring = state[3:]
        
    selection_score = np.sum(prob_scoring, axis=0) * unsolved_cells

    move = np.argmax(selection_score)

    return move

def greedy_with_penalty_policy(state):
    unsolved_cells = state[0] != -1
    prob_blowup = state[1]  
    prob_scoring = state[3:]
        
    selection_score = np.sum(prob_scoring, axis=0) / (0.001 + prob_blowup) * unsolved_cells

    move = np.argmax(selection_score)

    return move

def modified_greedy_with_penalty_policy(state):
    unsolved_cells = state[0] != -1
    prob_blowup = state[1]  
    prob_scoring = state[3:]
        
    selection_score = np.sum(prob_scoring, axis=0) / (0.02 + prob_blowup) * unsolved_cells

    move = np.argmax(selection_score)

    return move

def modified2_greedy_with_penalty_policy(state):
    unsolved_cells = state[0] != -1
    prob_blowup = state[1]  
    prob_scoring = state[3:]
        
    selection_score = np.sum(prob_scoring, axis=0) / (EPS + prob_blowup) * unsolved_cells

    move = np.argmax(selection_score)

    return move


def avoid_bomb_policy(state):
    unsolved_cells = state[0] != -1
    prob_blowup = state[1]  
        
    selection_score = 1 / (0.001 + prob_blowup) * unsolved_cells

    move = np.argmax(selection_score)

    return move

def entropy_policy(state):
    unsolved_cells = state[0] != -1
    prob_bomb = state[1]
    prob_num = state[2:]
        
    entropy = -np.sum(np.multiply(prob_num, np.log2(prob_num+EPS)), axis=0)
    selection_score = (entropy+0.001)/(prob_bomb+EPS) * unsolved_cells

    move = np.argmax(selection_score)

    return move

def modified_entropy_policy(state):
    unsolved_cells = state[0] != -1
    prob_bomb = state[1]
    prob_num = state[2:]
        
    entropy = -np.sum(np.multiply(prob_num, np.log2(prob_num+EPS)), axis=0)
    selection_score = (entropy+0.1)/(prob_bomb+0.02) * unsolved_cells

    move = np.argmax(selection_score)

    return move



class RLPolicy:
    def __init__(self, model_path, deterministic=False):
        self.model = VanillaPolicyGradient.load_from_checkpoint(model_path)
        self.model.eval()
        self.deterministic = deterministic

    def __call__(self, state):
        return self.model.get_action(state, self.deterministic)


if __name__ == "__main__":

    policies = {}

    for file in glob("logs/lightning_logs/version_28/checkpoints/non_det*.ckpt") + glob("logs/lightning_logs/version_29/checkpoints/*.ckpt"):
        groups = re.search("version_(\d+)*.*?epoch=(\d+)", file)
        version = groups.group(1)
        epoch = groups.group(2)

        # name = f"non_det_ver{version}_epoch{epoch}"
        # policies[name] = RLPolicy(file, deterministic=False)

        name = f"det_ver{version}_epoch{epoch}"
        policies[name] = RLPolicy(file, deterministic=True)

    policies["greedy"] = greedy_policy
    policies["greedy_with_penalty"] = greedy_with_penalty_policy
    policies["entropy"] = entropy_policy
    policies["avoid_bomb"] = avoid_bomb_policy
    policies["modified_greedy_with_penalty"] = modified_greedy_with_penalty_policy
    policies["modified_entropy"] = modified_entropy_policy
    policies["modified2_greedy_with_penalty"] = modified2_greedy_with_penalty_policy

    env = gym.make('VoltorbFlip-v0')

    for policy_name, policy in tqdm(policies.items()):
        path = "results/%s/" % policy_name
        os.makedirs(path, exist_ok=True)

        finished_games = 0
        if os.path.exists(path + GAME_LOG_FILE):
            with open(path + GAME_LOG_FILE, 'r') as fp:
                for count, line in enumerate(fp):
                    pass
            finished_games = count + 1

        

        for game_num in tqdm(range(finished_games, NUM_GAMES)):
            state = env.reset()
            num_moves = 0
            done = False
            cum_entropy = 0
            cum_prob_bomb = 0
            
            while not done:
                # collect_statistics(env.visible_board, env.board, env.solver.row_constraints, env.solver.col_constraints)
                num_moves += 1

                # Determine the best move
                action = policy(state)

                next_state, reward, done, _ = env.step(action)

                y = action // 5
                x = action % 5
                
                level, p0, p1, p2, p3 = state[:,y,x]
                cum_prob_bomb += (1 - cum_prob_bomb) * p0
                num_probs = np.array([p1, p2, p3])
                cum_entropy += -np.sum(np.multiply(num_probs, np.log2(num_probs+EPS)))
                cell = int(np.argmax(next_state[1:,y,x]))
                append_to_csv(path + MOVE_LOG_FILE, [level, num_moves, p0, p1, p2, p3, cum_prob_bomb, cum_entropy, cell])

                state = next_state

                if done:
                    if reward <= 0:
                        append_to_csv(path + GAME_LOG_FILE, [level, num_moves, cum_prob_bomb, cum_entropy, 0])
                    else:
                        append_to_csv(path + GAME_LOG_FILE, [level, num_moves, cum_prob_bomb, cum_entropy,1])
