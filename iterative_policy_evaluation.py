'''
table 하나를 이용한 반복 정책 평가
'''
import numpy as np
import pandas as pd
import time
import math
import random

np.random.seed(0)
env = Environment()
agent = Agent()
gamma = 0.9
v_table = np.zeros((env.reward_shape[0], env.reward_shape[1]))
k = 1
epsilon = 0.000001 
while (True):
    delta = 0
    temp_v = copy.deepcopy(v_table)
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            G = 0
            for action in range(len(agent.action)):
                agent.set_pos([i,j])
                observation, reward, done = env.move(agent, action)
                G += agent.select_action_pr[action] * (reward + gamma * v_table[observation[0], observation[1]])

            v_table[i,j] = G
    # 계산 전과 후의 가치 차이 계산
    delta = np.max([delta, np.max(np.abs(temp_v - v_table))])
    k += 1
    if delta < epsilon:
        break
