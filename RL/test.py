import gym
import time

env = gym.make("CartPole-v0", render_mode='human')

env.reset()
env.render()
time.sleep(100)
