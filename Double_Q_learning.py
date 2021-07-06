np.random.seed(0)
env = Environment()
agent = Agent()
gamma = 0.9


Q1_table = np.random.rand(env.reward.shape[0], env.reward.shape[1], len(agent.action))
Q2_table = np.random.rand(env.reward.shape[0], env.reward.shape[1], len(agent.action))

Q1_table[2,2,:] = 0
Q2_table[2,2,:] = 0

max_episode = 10000
max_step = 10

print("Start Double Q-learning")
alpha = 0.1
epsilon = 0.3

# 각 에피소드에 대해 반복
for epi in tqdm(range(max_episode)):
    i, j = 0, 0
    agent.set_pos([i,j])
    for k in range(max_step):
        pos = agent.get_pos()
        # Q1과 Q2로부터 a를 선택
        Q = Q1_table + Q2_table
        action = e_greedy(Q,agent,epsilon)
        # 행동 a를 취한 후 보상 r과 다음 상태 s'를 관측
        observation, reward, done = env.move(agent, action)
        # s'에서 Target Policy 행동 a'를 선택 (e.g. greedy)
        p = np.random.random()
        if p < 0.5:
            next_action = greedy(Q1_table, agent, epsilon)
            Q1_table[pos[0],pos[1],action] += alpha * (reward + gamma * Q2_table[observation[0], observation[1], next_action] - Q1_table[pos[0], pos[1], action])
        else:
            next_action = greedy(Q2_table, agent, epsilon)
            Q2_table[pos[0],pos[1],action] += alpha * (reward + gamma * Q1_table[observation[0],observation[1],next_action] - Q2_table[pos[0],pos[1],action])

        if done == True:
            break

# 학습된 정책에서 최적 행동 추출
optimal_policy = np.zeros((env.reward.shape[0], env.reward.shape[1]))
for i in range(env.reward.shape[0]):
    for j in range(env.reward.shape[1]):
        optimal_policy[i,j] = np.argmax(Q1_table[i,j,:] + Q2_table[i,j,:])


print("Double Q-learning: Q1(s,a)")
show_q_table(np.round(Q1_table,2),env)
print("Double Q-learning: Q2(s,a)")
show_q_table(np.round(Q2_table,2),env)
print("Double Q-learning: Q(s,a)")
show_q_table(np.round(Q1_table + Q2_table, 2), env)
print("Double Q-learning: optimal policy")
show_policy(optimal_policy, env) 
