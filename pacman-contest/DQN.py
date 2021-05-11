import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from capture import runGames, readCommand

# Hyper Parameters
BATCH_SIZE = 512
LR = 0.01  # learning rate
EPSILON = 0.6  # greedy policy
GAMMA = 0.8  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 65536
N_ACTIONS = 5
N_STATES = 39
TRAINING_SIZE = 10
TEST_SIZE = 0


class Net(nn.Module):
    def __init__(self, FEATURE_SIZE, ACTION_SIZE):
        super(Net, self).__init__()
        # three layers nn, hidden layer 50 nodes
        self.fc1 = nn.Linear(FEATURE_SIZE, 50)  # N_STATES: size of input layer
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, ACTION_SIZE)  # N_ACTIONS: size of output layer
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class DQN(object):
    m_loaded = False

    def __init__(self, FEATURE_SIZE, ACTION_SIZE, eval_file, tar_file, mem_file):
        self.eval_net, self.target_net = Net(FEATURE_SIZE, ACTION_SIZE), Net(FEATURE_SIZE,
                                                                             ACTION_SIZE)  # eval: evaluation net; target: realistic net
        self.load(eval_file, tar_file, mem_file)
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        if not self.m_loaded:
            self.memory = np.zeros((MEMORY_CAPACITY, 2 * N_STATES + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
            action = action
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, eval_file, tar_file, mem_file):
        torch.save(self.eval_net.state_dict(), eval_file)
        torch.save(self.target_net.state_dict(), tar_file)
        self.memory.tofile(mem_file)

    def load(self, eval_file, tar_file, mem_file):
        if eval_file is not None and os.path.isfile(eval_file):
            self.eval_net.load_state_dict(torch.load(eval_file))
        if tar_file is not None and os.path.isfile(tar_file):
            self.target_net.load_state_dict(torch.load(tar_file))
        if mem_file is not None and os.path.isfile(mem_file):
            f = open(mem_file, "r+")
            self.memory = np.fromfile(mem_file, dtype=np.float64)
            self.memory.shape = (MEMORY_CAPACITY, 2 * N_STATES + 2)
            self.m_loaded = True
            f.close()

    def write_counter(self, cnt_file_name):
        f = open(cnt_file_name, "w+")
        lines = [str(self.learn_step_counter), str(self.memory_counter)]
        for line in lines:
            f.write(line + "\n")
        f.close()

    def read_counter(self, cnt_file_name):
        if os.path.isfile(cnt_file_name):
            f = open(cnt_file_name, "r+")
            line = f.readline()
            self.learn_step_counter = int(line)
            line = f.readline()
            self.memory_counter = int(line)
            f.close()



if __name__ == "__main__":

    for i_episode in range(TRAINING_SIZE):
        print "PLAYING TRAINING GAME " + str(i_episode) + ":"
        arguments = ["-l", "RANDOM",
                     "-r", "kizunaTeam",
                     "-b", "baselineTeam", "-q",
                     "--time", "80000"]
        options = readCommand(arguments)
        games = runGames(**options)
        print "\n\n"

    EPSILON = 1.0
    for i_episode in range(TEST_SIZE):
        print "PLAYING TEST GAME " + str(i_episode) + ":"
        arguments = ["-l", "RANDOM",
                     "-r", "kizunaTeam",
                     "-b", "baselineTeam"]
        options = readCommand(arguments)
        games = runGames(**options)
        print "\n\n"

    # print('\nCollecting experience...')
    # for i_episode in range(TRAINING_SIZE):
    #
    #     s = env.reset()
    #     print s
    #     ep_r = 0
    #     while True:
    #         env.render()
    #         a = dqn.choose_action(s)
    #
    #         # take action
    #         s_, r, done, info = env.step(a)
    #
    #         # modify the reward
    #         x, x_dot, theta, theta_dot = s_
    #         r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    #         r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    #         r = r1 + r2
    #
    #         dqn.store_transition(s, a, r, s_)
    #
    #         ep_r += r
    #         if dqn.memory_counter > MEMORY_CAPACITY:
    #             dqn.learn()
    #             if done:
    #                 print('Ep: ', i_episode,
    #                       '| Ep_r: ', round(ep_r, 2))
    #
    #         if done:
    #             break
    #         s = s_
    #         EPSILON *= 0.9
    #
    # dqn.eval_net.save("eva.pkl")
