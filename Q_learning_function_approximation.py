env = Environment()
agent = Agent([0,0])
gamma = 0.9
np.random.seed(0)

w = np.random.rand(len(agent.action), env.reward.shape[0])
w -= 0.5
max_episode = 10000
max_step = 100
alpha = 0.01

for epi in tqdm(range(max_episode)):
    i,j = 0,0
    agent.set_pos([i,j])
    # 에피소드의 각 스텝에 대해 반복
    for k in range(max_step):
        pos = agent.get_pos()
        action = np.zeros(4)
        for act in range(len(agent.action)):
            action[act] = w[act,0] + w[act,1]*pos[0] + w[act,2]*pos[1]
        pr = np.zeros(4)
        for i in range(len(agent.action)):
            pr[i] = np.exp(action[i]) / np.sum(np.exp(action[:]))
        action = np.random.choice(range(0,len(agent.action)), p=pr)
        observation, reward, done = env.move(agent,action)
        next_act = np.zeros(4)
        for act in range(len(agent.action)):
            next_act[act] = np.dot(w[act,1:], observation) + w[act,0]

        best_action = np.argmax(next_act)
        now_q = np.dot(w[action,1:],pos) + w[action,0]
        next_q = np.dot(w[best_action,1:],pos) + w[best_action,0]

        # update
        w[action, 0] += alpha*(reward + gamma * next_q - now_q)
        w[action, 1] += alpha*(reward + gamma * next_q - now_q) * pos[0]
        w[action, 2] += alpha*(reward + gamma * next_q - now_q) * pos[1]

        if done == True:
            break 
