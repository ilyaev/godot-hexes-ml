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
import math
import atexit
import sys


class Gym:
    arena = Hex.Arena()
    data = dict()
    metrics = dict()
    session = dict({"name": "", "rounds": 0, "turns": 0,
                    "acts": 0, "won": 0, "save_per_turns": 500, 'generate_turn_graph': 'false'})
    model_file = ''

    def __init__(self):
        self.init_session()
        print('START_SESSION:', self.session)
        self.metrics = {'round': 0, 'turn': 1, 'act': 0,
                        'learn': 0, 'won': 0}
        self.new_round()
        self.net = DQNNet(len(self.arena.regions))
        self.model_file = 'models/' + self.session['name'] + '.save'
        if os.path.exists(self.model_file):
            self.net.load_state_dict(torch.load(self.model_file))
            self.net.eval()

    def init_session(self):
        lines = open('./names.csv').read().splitlines()
        self.session['name'] = random.choice(lines)
        cfgs = open('./.session').read().splitlines()
        for i in range(len(cfgs)):
            pair = cfgs[i].split('=')
            self.session[pair[0]] = pair[1]
        self.session['save_per_turns'] = int(self.session['save_per_turns'])

    def new_round(self):
        with open('models/map.json') as f:
            self.data = json.load(f)
        self.arena.load_from_file(os.getcwd().replace(
            '/env', '') + '/models/map.json')
        self.arena.new_game()
        self.metrics['round'] += 1
        self.session['rounds'] += 1
        print('NEW_ROUND:', self.metrics)

    def build_available_actions(self):
        actions = []
        if self.arena.selected_region < 0:
            for i in range(len(self.arena.regions)):
                if self.arena.regions[i].country_id == self.arena.active_player and self.arena.regions[i].population > 1:
                    avec = self.arena.regions[i].adjacency_vector
                    for j in range(len(avec)):
                        if avec[j] == 1 and self.arena.regions[j].country_id != self.arena.active_player:
                            df = self.arena.regions[j].population - \
                                self.arena.regions[i].population
                            if df <= 0:
                                actions.append(i)
            actions = actions + [self.arena.regions_count]
        else:
            avec = self.arena.regions[self.arena.selected_region].adjacency_vector
            for i in range(len(avec)):
                if avec[i] == 1 and self.arena.regions[i].country_id != self.arena.active_player:
                    df = self.arena.regions[i].population - \
                        self.arena.regions[self.arena.selected_region].population
                    if df <= 0:
                        actions.append(i)
            if len(actions) == 0:
                actions = actions + [self.arena.regions_count]

        return actions

    def evaluate(self):
        current_player = self.arena.active_player
        s = self.build_input()
        available_actions = self.build_available_actions()
        a = self.net.act(s, available_actions)
        self.metrics['act'] += 1
        self.session['acts'] += 1
        end_turn = False

        if a == self.arena.regions_count:
            end_turn = True
            if self.arena.active_player == self.arena.max_players - 1:
                self.metrics['turn'] += 1
                self.session['turns'] += 1
            self.metrics['learn'] = math.ceil(self.metrics['learn'])
            print('NEXT_PLAYER:' + str(self.arena.active_player),
                  self.metrics, self.arena.scores, self.arena.get_country_sizes())
            self.metrics['act'] = 0
            self.metrics['learn'] = 0
            self.metrics['won'] = 0

        r = self.arena.act(a, end_turn)
        if self.session['generate_turn_graph'] == 'true':
            graph_file = 'graphs/' + self.session['name'] + '_' + "{:010d}".format(self.session['acts']) + '_r' + str(
                self.session['rounds']) + '_t' + str(self.metrics['turn']) + '_c' + str(current_player) + '.png'
            graph_title = self.session['name'] + '           r:' + str(self.session['rounds']) + ' t:' + str(
                self.metrics['turn']) + ' c:' + str(current_player) + ' a:' + str(self.metrics['act'])
            self.arena.draw_country_network(
                current_player, graph_file, graph_title)
        if r > 0:
            # print('learn: ', r, 'a:', a)
            self.metrics['learn'] += r
            self.metrics['won'] += 1
            self.session['won'] += 1
        s1 = self.build_input()
        self.net.learn(s, a, s1, r)
        if end_turn == True:
            self.check_for_new_round()

    def check_for_new_round(self):
        eco = 0
        min_score = 3
        country_sizes = self.arena.get_country_sizes()
        for i in range(self.arena.max_players):
            if country_sizes[i] < min_score:
                eco += 1
        if eco >= self.arena.max_players - 1:
            self.new_round()

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


@atexit.register
def save_session():
    print("END_SESSION:", gym.session)


if __name__ == '__main__':
    gym = Gym()
    i = 0
    while True:
        try:
            if i % gym.session['save_per_turns'] == 0 and i > 0:
                torch.save(gym.net.state_dict(), gym.model_file)
                print('SAVE: ' + str(i))
            i = i + 1
            gym.evaluate()
        except ER:
            sys.exit()
            pass
