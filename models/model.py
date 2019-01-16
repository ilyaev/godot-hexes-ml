import json
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state',
                                       'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNNet(nn.Module):

    epsilon = 0.1
    gamma = 0.9
    threshold = 0.000001
    max_learning_iterations = 100
    s1 = list()
    s0 = list()
    r0 = 0
    a0 = 0
    a1 = 0
    criterion = nn.MSELoss()
    memory = ReplayMemory(10000)
    actions_len = 0
    batch_size = 10

    def __init__(self, regions_count):
        super(DQNNet, self).__init__()
        input_count = regions_count*regions_count + regions_count * 2
        output_count = 1 + regions_count
        self.fc1 = nn.Linear(input_count, input_count)
        self.fc2 = nn.Linear(input_count, output_count)
        self.actions_len = output_count
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, s, av):
        a = 0
        if random.random() <= self.epsilon:
            a = av[random.randint(0, len(av) - 1)]
        else:
            actions = self.forward(torch.tensor(s).float()).tolist()
            for i in range(self.actions_len):
                if not(i in av):
                    actions[i] = -1000
            a = actions.index(
                max(actions))

        self.s0 = self.s1
        self.a0 = self.a0
        self.s1 = s
        self.a1 = a
        return a

    def learn(self, s, a, s1, r):
        batch = [Transition(s, a, s1, r)] + self.memory.sample(
            min(self.batch_size, len(self.memory)))

        for bi in range(len(batch)):
            one = batch[bi]
            pred = self.forward(torch.tensor(one.next_state).float()).tolist()
            q = one.reward + self.gamma * max(pred)
            pred[one.action] = q

            input = torch.tensor(one.state).float()
            target = torch.tensor(pred).float()

            for idx in range(self.max_learning_iterations):
                self.optimizer.zero_grad()
                output = self.forward(input)
                # print(output[self.a1].data, target[self.a1].data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                # print('Loss: ', loss.data * 1000)
                if loss.data < self.threshold:
                    break

        self.memory.push(s, a, s1, r)
        return r
        pass
