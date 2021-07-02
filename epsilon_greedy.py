## epsilon greedy for the grid search problem

def e_greedy(Q_table, agent, epsilon):
    pos = agent.get_pos()
    greedy_action = np.argmax(Q_table[pos[0],pos[1],:])
    pr = np.zeros(4)
    for i in range(len(agent.action)):
        if i == greedy_action:
            pr[i] = 1-epsilon+epsilon/len(agent.action)
        else:
            pr[i] = epsilon/len(agent.action)
    return np.random.choice(range(0, len(agent.action)), p=pr)

def greedy(Q_tabe, agent, epsilon):
    pos = agent.get_pos()
    return np.argmax(Q_table[pos[0],pos[1],:]) 
