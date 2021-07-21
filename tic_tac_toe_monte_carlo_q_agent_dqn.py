def move(self, p1, p2, player):
    if player == 1:
        pos = p1.select_action(env,player)
    else:
        pos = p2.select_action(env, player)

    self.board_a[pos] = player
    if self.print:
        print(player)
        self.print_board()

    self.end_check(player)
    return self.reward, self.done

class Environment:
    def __init__(self):
        self.board_a = np.zeros(9)
        self.done = False
        self.reward = 0
        self.winner = 0
        self.print = False

    def move(self, p1, p2, player):
        # 각 플레이어가 선택한 행동을 표시하고 게임 상태를 판단
        if player == 1:
            pos = p1.select_action(env, player)
        else:
            pos = p2.select_action(env, player)

        self.board_a[pos] = player
        self.print_board()
        self.end_check(player)
        return self.reward, self.done

    def get_action(self):
        observation = []
        for i in range(9):
            if self.board_a[i] == 0:
                observation.append(i)
        return observation

    def end_check(self, player):
        end_condition = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for line in end_condition:
            if self.board_a[line[0]] == self.board_a[line[1]] and self.board_a[line[1]] == self.board_a[line[2]] and self.board_a[line[0]] != 0:
                self.done = True
                self.reward = player
                return
        observation = self.get_action()
        if len(observation) == 0:
            self.done = True
            self.reward = 0
        return

    def print_board(self):
        pass


'''
5 players will be made
1. Human Player
2. Random Player
3. Monte-Carlo player
4. Q-learning Player
5. DQN player
'''

### human player
class Human_player:
    def __init__(self):
        self.name = "Human player"

    def select_action(self, env, player):
        while True:
            # 가능한 행동들 조회후 표기
            available_action = env.get_action()
            print("possible actions = {}".format(available_action))
            print("+---+---+---+")
            print("+ 0 + 1 + 2 +")
            print("+---+---+---+")
            print("+ 3 + 4 + 5 +")
            print("+---+---+---+")
            print("+ 6 + 7 + 8 +")
            print("+---+---+---+")

            action = input("select action(human) : ")
            action = int(action)

            if action in available_action:
                return action

            else:
                print("You selected wrong action")
        return


### random player
class Random_player:
    def __init__(self):
        self.name = "Random player"

    def select_action(self, env, player):
        available_action = env.get_action()
        action = np.random.randint(len(available_action))
        return available_action[action]


## 게임 진행 함수
p1 = Human_player()
p2 = Random_player()

auto = False # 자동 or 수동
games = 100
print("p1 player : {}".format(p1.name))
print("p2 player : {}".format(p2.name))

p1_score = 0
p2_score = 0
draw_score = 0
if auto:
    for j in range(games):
        nnp.random.seed(j)
        env = environment()
        for i in range(10000):
            reward, done = env.move(p1,p2,(-1)**i)
            if done == True:
                if reward == 1:
                    print("j = {} winner is p1({})".format(j, p1.name))
                    p1_score += 1
                elif reward == -1:
                    print("j = {} winner is p2({})".format(j, p2.name))
                    p2_score += 1
                else:
                    print("j = {} draw".format(j))
                    draw_score += 1
                break
else:
    while True:
        env = environment()
        env.print = True
        for i in range(10000):
            reward, done = env.move(p1,p2,(-1)**i)
            if done == True:
                if reward == 1:
                    print("winner is p1({})".format(p1.name))
                    p1_score += 1
                elif reward == -1:
                    print("winner is p2({})".format(p2.name))
                    p2_score += 1
                else:
                    print("draw")
                    draw_score += 1
                break
            print("final result")
            env.print_board()
            print("final result")
            env.print_board()
            answer = input("More Game? (y/n)")
print("p1({}) = {} p2({}) = {} draw = {}".format(p1.name, p1_score, p2.name, p2_score.draw_score))



class Monte_Carlo_player:
    def __init__(self):
        self.name = "MC Player"
        self.num_playout = 1000
    def select_action(self, env, player):
        available_action = env.get_action()
        V = np.zeros(len(available_action))
        for i in range(len(available_action)):
            for j in range(self.num_playout):
                temp_env = copy.deepcopy(env)
                self.playout(temp_env, available_action[i], player)
                if player == temp_env.reward:
                    V[i] += 1
        return available_action[np.argmax(V)]


    def playout(self, temp_env, action, player):
        temp_env.board_a[action] = player
        temp_env.end_check(player)
        if temp_env.done == True:
            return
        else:
            # switch player
            player = -player
            available_action = temp_env.get_action()
            action = np.random.randint(len(available_action))
            self.playout(temp_env, available_action[action], player)



## Q-learning player
class Q_learning_player:
    def __init__(self):
        self.name = "Q_player"
        # Q-table을 딕셔너리로 정의
        self.qtable = {}
        # epsilon-greedy 계수 정의
        self.epsilon = 1

        # learning rate 정의
        self.learning_rate = 0.1
        self.gamma = 0.9

    # policy 에 따라 상태에 맞는 행동을 선택
    def select_action(self, env, player):
        action = self.policy(env)
        return action

    def policy(self, env):
        available_action = env.get_action()

        qvalues = np.zeros(len(available_action))

        # 행동 가능한 상태의 Q-value를 조사
        for i, act in enumerate(available_action):
            key = (tuple(env.board_a), act)
            # 현재 상태를 경험한 적이 없다면 딕셔너리에 추가 (Q-value = 0)
            if self.qtable.get(key) == None:
                self.qtable[key] = 0
            # 행동 가능한 상태의 Q-value 저장
            qvalues[i] = self.qtable.get(key)

        # epsilon-greedy
        greedy_action = np.argmax(qvalues)
        pr = np.zeros(len(available_action))

        # check if there are multiple max Q-values
        double_check = (np.where(qvalues==np.max(qvalues), 1, 0))

        if np.sum(double_check) > 1:
            double_check = double_check / np.sum(double_check)
            greedy_action = np.random.choice(range(0, len(double_check)), p=double_check)

        # epsilon-greedy 행동들의 선택 확률 계산
        pr = np.zeros(len(available_action))
        for i in range(len(available_action)):
            if i == greedy_action:
                pr[i] = 1 - self.epsilon + self.epsilon / len(available_action)
            else:
                pr[i] = sef.epsilon / len(available_action)
        action = np.random.choice(range(0,len(available_action)), p=pr)
        return available_action[action]


def learn_qtable(self, board_backup, action_backup, env, reward):
    # 현재 상태와 행동을 키로 저장
    key = (board_backup, action_backup)
    if env.done == True:
        self.qtable[key] += self.learning_rate * (reward - self.qtable[key])
    else:
        available_action = env.get_action()
        qvalues = np.zeros(len(available_action))
        for i, act in enumerate(available_action):
            next_key = (tuple(env.board_a), act)
            # 다음 상태를 경험한 적이 없다면 (딕셔너리에 없다면) 딕셔너리에 추가 (Q-value = 0)
            if self.qtable.get(next_key) == None:
                self.qtable[next_key] = 0
            qvalues[i] = self.qtable.get(next_key)

        # maxQ 조사
        maxQ = np.max(qvalues)
        # 게임 진행중일때 학습
        self.qtable[key] += self.learning_rate * (reward + self.gamma * maxQ - self.qtable[key])


## training the Q learning agent
p1_Qplayer = Q_learning_player()
p2_Qplayer = Q_learning_player()

p1_Qplayer.epsilon = 0.5
p2_Qplayer.epsilon = 0.5
p1_score = 0
p2_score = 0
draw_score = 0
max_learn = 100000

for j in range(max_learn):
    np.random.seed(j)
    env = environment()
    for i in range(10000):
        # p1 행동 선택
        player = 1
        pos = p1_Qplayer.policy(env)

        # 현재 상태 s, 행동 a를 저장
        p1_board_backup = tuple(env.board_a)
        p1_action_backup = pos
        env.board_a[pos] = player
        env.end_check(player)

        if env.done == True:
            if env.reward == 0:
                p1_Qplayer.learn_qtable(p1_board_backup, p1_action_backup, env, 0)
                p2_Qplayer.learn_qtable(p2_board_backup, p2_action_backup, env, 0)
                draw_score += 1
                break
            # p1이 이겼으므로 보상 +1로 학습
            # p2가 졌으므로 보상 -1로 학습
            else:
                p1_Qplayer.learn_qtable(p1_board_backup, p1_action_backup, env, 1)
                p2_Qplayer.learn_qtable(p2_board_backup, p2_action_backup, env, -1)
                p1_score += 1
                break

            # 게임이 끝나지 않았다면 p2의 Q-table을 학습 (게임 시작 직후에는 p2는 학습할 수 없음)
            if i != 0:
                p2_Qplayer.learn_qtable(p2_board_backup, p2_action_backup, env, -0.01)

            # p2 행동 선택
            player = -1
            pos = p2_Qplayer.policy(env)
            p2_board_backup = tuple(env.board_a)
            p2_action_backup = pos
            env.board_a[pos] = player
            env.end_check(player)

            if env.done == True:
                # 비겼으면 보상 0으로 p1, p2 플레이어 학습
                if env.reward == 0:
                    p1_Qplayer.learn_qtable(p1_board_backup, p1_action_backup, env, 0)
                    p2_Qplayer.learn_qtable(p2_board_backup, p2_action_backup, env, 0)
                    draw_score += 1
                    break
                # p2가 이겼으므로 보상 +1로 학습
                # p1이 졌으므로 보상 -1로 학습
                else:
                    p1_Qplayer.learn_qtable(p1_board_backup, p1_action_backup, env, -1)
                    p2_Qplayer.learn_qtable(p2_board_backup, p2_action_backup, env, -1)
                    p2_score += 1
                    break
            # 게임이 끝나지 않았다면 p1의 Q-table 학습
            p1_Qplayer.learn_qtable(p1_board_backup, p1_action_backup, env, -0.01)

    if j%1000 == 0:
        print("j = {} p1 = {} p2 = {} draw = {}".format(j, p1_score, p2_score, draw_score))
print("p1 = {} p2 = {} draw = {}".format(p1_score, p2_score, draw_score))
print("end train")


## DQN Player
from keras.models import Sequential
from keras.optimizers import SGD
from keras import metrics
from keras.layers import Dense, Flatten, Conv2D
import time

class DQN_player:
    def __init__(self):
        self.name = "DQN_player"
        self.epsilon = 1
        self.learning_rate = 0.1
        self.gamma = 0.9
        # 두 개의 신경망을 생성
        self.main_network = self.make_network()
        self.target_network = self.make_network()

        # 메인 신경망의 가중치를 타깃 신경망의 가중치로 복사
        self.copy_network()
        self.count = np.zeros(9)
        self.win = np.zeros(9)
        self.begin = 0

    def make_network(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, (3,3), padding='same', activation = 'relu', input_shape=(3,3,2)))
        self.model.add(Conv2D(32, (3,3), padding='same', activation = 'relu'))
        self.model.add(Conv2D(64, (3,3), padding='same', activation = 'relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation = 'tanh'))
        self.model.add(Dense(128, activation = 'tanh'))
        self.model.add(Dense(64, activation = 'tanh'))
        self.model.add(Dense(9))
        print(self.model.summary())
        self.model.compile(optimizer = SGD(lr=0.01), loss='mean_squared_error', metrics=['mse'])
        return self.model

    def copy_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def state_convert(self, board_a):
        d_state = np.full((3,3,2), 0.1)
        for i in range(9):
            if board_a[i] == 1:
                d_state[i//3,i%3,0] = 1
            elif board_a[i] == -1:
                d_state[i//3,i%3,1] = 1
            else:
                pass
        return d_state

    def select_action(self, env, player):
        action = self.policy(env)
        return action


    def policy(self, env):
        available_state = env.get_action()
        state_2d = self.state_convert(env.board_a)
        x = np.array([state_2d], dtype=np.float32).astype(np.float32)
        qvalues = self.main_network.predict(x)[0,:]

        available_state_qvalues = qvalues[available_state]
        greedy_action = np.argmax(available_state_qvalues)
        double_check = (np.where(qvalues == np.max(available_state[greedy_action]), 1, 0))

        if np.sum(double_check) > 1:
            double_check = double_check / np.sum(double_check)
            greedy_action = np.random.choice(range(0,len(double_check)), p=double_check)

        pr = np.zeros(len(available_state))
        for i in range(len(available_state)):
            if i == greedy_action:
                pr[i] = 1-self.epsilon + self.epsilon / len(available_state)
            else:
                pr[i] = self.epsilon / len(available_state)
        action = np.random.state(range(0, len(available_state)), p=pr)
        return available_state[action]

    '''
    1. 1차원 입력을 2차원으로 변환
    2. qvalues를 메인 신경망으로부터 구하는 방법
    3. 학습은 메인 신경망이 대상
    4. Max Q-value는 타깃 신경망으로 계산
    5. 메인 신경망의 학습을 위한 yhat은 메인 신경망의 출력과 Max Q-value를 이용해 Q(s,a)를 계산하고 a의 위치에 학습된 값을 바꿔치기함
    '''

    def learn_dqn(self, board_backup, action_backup, env, reward):
        # 입력을 2차원으로 변환 후, 메인 신경망으로 Q-value를 계산
        new_state = self.state_convert(board_backup)
        x = np.array([new_state], dtype=np.float32).astype(np.float32)
        qvalues = self.main_network.predict(x)[0,:]
        before_action_value = copy.deepcopy(qvalues)
        delta = 0
        if env.done == True:
            qvalues[action_backup] = reward
            y = np.array([qvalues], dtype=np.float32).astype(np.float32)
            # 생성된 정답 데이터로 메인 신경망을 학습
            self.main_network.fit(x,y,epochs=10,verbose=0)
        else:
            # 게임이 진행 중일 때 신경망의 학습을 위한 정답 데이터를 생성
            # 현재 상태에서 max Q-value를 계산
            new_state = self.state_convert(env.board_a)
            next_x = np.array([new_state], dtype=np.float32).astype(np.float32)
            next_qvalues = self.target_network.predict(next_x)[0,:]
            available_state = env.get_action()
            maxQ = np.max(next_qvalues[available_state])
            delta = self.learning_rate * (reward + self.gamma * maxQ - qvalues[action_backup])
            qvalues[action_backup] += delta
            y = np.array([qvalues], dtype=np.float32).astype(np.float32)
            self.main_network.fit(x,y,epochs=10,verbose=0)
