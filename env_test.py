import sys, os
import time
import numpy as np
import random
import traceback
import argparse
import gym
import gym_sfm.envs.env as envs

parser = argparse.ArgumentParser()
parser.add_argument('--map', help='Specify map setting folder.', default='demo')
parser.add_argument('-tl', '--time_limit', help='Specify env time limit(sec).', type=int, default=1800)
parser.add_argument('-mt', '--max_t', type=int, default=1800)
parser.add_argument('-mepi', '--max_episodes', type=int, default=5)
args = parser.parse_args()

env = gym.make('gym_sfm-v0', md=args.map, tl=args.time_limit)

for i_episode in range(args.max_episodes):
    observation = env.reset()
    # env.agent.pose = np.array([0.0, 0.0])
    done = False
    epidode_reward_sum = 0
    # start = time.time()
    for t in range(args.max_t):
        action = np.array([0, 0], dtype=np.float64)
        observation, reward, done, _ = env.step(action)
        # print(observation)
        env.render()
        # print(done)
        # if not done :
        #     # env.close()
        #     break
    env.close()
    # end = time.time()
    # print(end - start)
print('Finished all episode.')