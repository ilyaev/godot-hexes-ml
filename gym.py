import env.hex as Hex
from models.model import DQNNet
import json
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os


class Gym:
    arena = Hex.Arena()
    data = dict()

    def __init__(self):
        with open('models/map.json') as f:
            self.data = json.load(f)
        self.arena.load_from_file(os.getcwd().replace(
            '/env', '') + '/models/map.json')
        self.arena.new_game()
        self.net = DQNNet(len(self.arena.regions))
        if os.path.exists('models/map.save'):
            print('RESTORE')
            self.net.load_state_dict(torch.load('models/map.save'))
            self.net.eval()

    def evaluate(self):
        s = self.build_input()
        a = self.net.act(s)
        end_turn = False
        if a == self.arena.regions_count:
            end_turn = True
            print('END_TURN -> ' + str(self.arena.active_player))
        r = self.arena.act(a, end_turn)
        s1 = self.build_input()
        self.net.learn(s, a, s1, r)

    def build_input(self):
        data = self.data
        input = []
        for i in range(len(data['full'])):
            region = self.arena.regions[i]
            row = [] + data['full'][i]
            for k in range(len(row)):
                target = self.arena.regions[k]
                if row[k] == 0:
                    row[k] = -1
                else:
                    row[k] = (target.population - region.population) * 0.2
            input = input + row

        selection_row = list(
            map(lambda x: 0, range(len(self.arena.regions))))
        if self.arena.selected_region != -1:
            selection_row[self.arena.selected_region] = 1

        country_row = list(
            map(lambda x: 0, range(len(self.arena.regions))))
        for i in range(len(self.arena.regions)):
            region = self.arena.regions[i]
            if region.country_id == self.arena.active_player:
                country_row[i] = 0.1 * region.population

        input = input + selection_row + country_row

        return input


if __name__ == '__main__':
    gym = Gym()
    i = 0
    while True:
        if i % 100 == 0:
            torch.save(gym.net.state_dict(), 'models/map.save')
            print('SAVE: ' + str(i))
        i = i + 1
        gym.evaluate()
