import numpy as np
import pandas as pd
import time
import math
from tqdm import tqdm

''' define environment class '''
class Environment:
    cliff = -3 # reward
    road = -1 # minmize no. of steps
    goal = 1

    goal_position = [2,2]

    reward_list = [[road,road,road],
                   [road,road,road],
                   [road,road,goal]]

    # reward list in string
    reward_list1 = [["road","road","road"],
                    ["road","road","road"],
                    ["road","road","goal"]]

    def __init__(self):
        self.reward = np.asarray(self.reward_list)

    def move(self, agent, action):
        done = False
        new_pos = agent.pos + agent.action[action]

        # check if the current destination is the goal
        if self.reward_list1[agent.pos[0]][agent.pos[1]] == "goal":
            reward = self.goal
            observation = agent.set_pos(agent.pos)
            done = True
        elif new_pos[0] < 0 or new_pos[0] >= self.reward.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.reward.shape[1]:
            reward = self.cliff
            observation = agent.set_pos(agent.pos)
            done = True
        else:
            observation = agent.set_pos(new_pos)
            reward = self.reward[observation[0], observation[1]]

        return observation, reward, done

''' define agent class '''
class Agent:
    action = np.array([[-1,0],[1,0],[0,-1],[0,1]])
    select_action_pr = np.array([0.25,0.25,0.25,0.25])

    def __init__(self, initial_position):
        self.pos = initial_position

    def set_pos(self,position):
        self.pos = position
        return self.pos

    def get_pos(self):
        return self.pos

def state_value_function(env, agent, G, max_step, now_step):
    gamma = 0.9
    if env.reward_list1[agent.pos[0]][agent.pos[1]] == 'goal':
        return env.goal
    # calculate the reward fo the last step
    # end of recursion
    if max_step == now_step:
        pos1 = agent.get_pos()
        for i in range(len(agent.action)):
            agent.set_pos(pos1)
            observation, reward, done = env.move(agent, i)
            G += agent.select_action_pr[i] * reward
        return G
    else:
        pos1 = agent.get_pos()
        for i in range(len(agent.action)):
            observation, reward, done = env.move(agent, i)
            G += agent.select_action_pr[i] * reward

            # check if agent left the grid or if the agent is at a wall position
            if done == True:
                if observation[0] < 0 or observation[0] >= env.reward.shape[0] or observation[1] < 0 or observation[1] >= env.reward.shape[1]:
                    agent.set_pos(pos1)
            next_v = state_value_function(env, agent, 0, max_step, now_step + 1)
            G += agent.select_action_pr[i] * gamma * next_v
            agent.set_pos(pos1) # set to initial position
        return G

''' run simple grid search '''
env = Environment()
agent = Agent([0,0])
max_step_number = 14
time_len = []
for max_step in tqdm(range(max_step_number), position = 0, leave = True):
    v_table = np.zeros((env.reward.shape[0], env.reward.shape[1]))
    start_time = time.time()
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            agent.set_pos([i,j])
            v_table[i,j] = state_value_function(env, agent, 0, max_step, 0)
    time_len.append(time.time()-start_time)
    print("max_step_number = {}, total_time = {}".format(max_step, np.round(time.time()-start_time,2)))
