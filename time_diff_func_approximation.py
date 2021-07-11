np.random.seed(1)

env = Environment()
agent = Agent()
gamma = 0.9
w = np.random.rand(env.reward.shape[0])
w -= 0.5

v_table = np.zeros((env.reward.shape[0], env.reward.shape[1]))
for i in range(env.reward.shape[0]):
    for j in range(env.reward.shape[1]):
        v_table[i,j] = w[0] + w[1]*i + w[2]*j
max_episode = 10000
max_step = 100
alpha = 0.01
epsilon = 0.3
print("start funciton approximation TD(0) prediction")
for epi in tqdm(range(max_episode)):
    delta = 0
    i,j = 0
    agent.set_pos([i,j])
    temp = 0
    for k in range(max_step):
        pos = agent.get_pos()
        action = np.random.randint(0,len(agent.action))
        observation, reward, done = env.move(agent.action)
        now_v, next_v = 0,0
        now_v = w[0] + np.dot(w[1:],pos)
        next_v = w[0] + np.dot(w[1:],observation)

        w[0] += alpha * (reward + gamma*next_v - now_v)
        w[1] += alpha * (reward + gamma*next_v - now_v) * pos[0]
        w[2] += alpha * (reward + gamma*next_v - now_v) * pos[1]

        if done == True:
            break

for i in range(env.reward.shape[0]):
    for j in range(env.reward.shape[1]):
        v_table[i,j] = w[0] + w[1]*i + w[2]*j

show_v_table(np.round(v_table,2), env) 
