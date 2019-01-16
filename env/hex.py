import random
import os
import json
import networkx as nx
import matplotlib.pyplot as plt


MAX_POPULATION = 8
INITIAL_POPULATION_PER_REGION = 4


class Region:

    country_id = -1
    id = -1
    adjacency_vector = []
    population = 0

    def __init__(self):
        pass


class Arena:

    regions = []
    regions_count = 0
    adjacency_matrix = []
    countries = 0
    country_regions = []
    country_scores = []
    selected_region = -1
    scores = []
    G = 0
    _op = 0

    active_player = 0
    max_players = 0
    round = 0

    def __init__(self):
        pass

    def _throw_dices(self, dices):
        result = 0
        for dice in range(dices):
            result = result + random.randint(1, 6)
        return dices  # debug

    def act(self, region_id, end_turn=False):
        if end_turn:
            self.end_turn()
            return 0

        if self.selected_region == -1:
            if self.regions[region_id].country_id != self.active_player or region_id == self.selected_region:
                return -1
            else:
                self.selected_region = region_id
                # print('SEL: ', region_id, 'CO:', self.active_player)
                return 0

        src_region_id = self.selected_region
        target_region_id = region_id

        src_region = self.regions[src_region_id]
        target_region = self.regions[target_region_id]

        if target_region.country_id == self.active_player:
            return -1

        if src_region.adjacency_vector[target_region_id] == 0:
            return -1

        # print('BAT: ', self.selected_region, '->', target_region_id)

        score_before = self.scores[src_region.country_id]

        src = self._throw_dices(src_region.population)
        target = self._throw_dices(target_region.population)

        if src > target:
            target_region.population = src_region.population - 1
            src_region.population = 1

            self.country_regions[target_region.country_id].remove(
                target_region_id)
            target_region.country_id = src_region.country_id
            self.country_regions[target_region.country_id].extend([
                target_region_id])
            self._calculate_scores()
        else:
            src_region.population = 1

        score_after = self.scores[src_region.country_id]
        score_diff = score_after - score_before
        self.selected_region = -1
        if score_diff > 0:
            return 1
        else:
            return 0

    def new_game(self):
        self._build_country_map()
        self._plant_population()
        self._calculate_scores()
        self.max_players = len(self.scores)
        # self.print_country(0)

    def end_turn(self):
        # print('--BEFORE')
        # self.print_country(self.active_player)
        tod = self.scores[self.active_player]
        rgs = self.country_regions[self.active_player]
        # print('RGS:', rgs)
        while tod > 0:
            idx = rgs[random.randint(
                0, len(rgs) - 1)]
            canbe = False
            for i in range(len(rgs)):
                if self.regions[rgs[i]].population < MAX_POPULATION:
                    canbe = True
            if not canbe:
                tod = 0
            if self.regions[idx].population < MAX_POPULATION:
                tod = tod - 1
                self.regions[idx].population = self.regions[idx].population + 1
            # print('tod: ', tod, idx, )
        # print('--AFTER')
        # self.print_country(self.active_player)
        self.active_player = self.active_player + 1
        if self.active_player >= self.max_players:
            self.active_player = 0
        self.selected_region = -1
        pass

    def draw_country_network(self, country_id, postfix=''):
        nodelist = []
        edgelist = []
        for region_id in self.country_regions[country_id]:
            nodelist.extend([region_id])
            for edge in self.G.edges(region_id):
                if self.country_regions[country_id].count(edge[0]) > 0 and self.country_regions[country_id].count(edge[1]) > 0:
                    edgelist.extend([edge])

        options = {
            'node_size': 300,
            'width': 1,
            'node_color': '#33cc33',
            'font_size': 10,
            'nodelist': list(nodelist),
            'edgelist': list(edgelist),
            'with_labels': True
        }
        plt.get_current_fig_manager().canvas.figure.clear()
        nx.draw_kamada_kawai(self.G, **options)  # best so far
        plt.savefig('env/graph_' + str(country_id) + postfix + '.png')

    def draw_network(self):
        options = {
            'node_size': 300,
            'width': 1,
            'node_color': '#EFEFEF',
            'font_size': 10,
            'with_labels': True
        }
        nx.draw_kamada_kawai(self.G, **options)  # best so far
        plt.get_current_fig_manager().canvas.figure.clear()
        plt.savefig('env/graph.png')

    def _get_connections(self, node, nodes):
        cid = 0
        connections = 0
        for i in self.adjacency_matrix[node]:
            self._op += 1
            if nodes.count(cid) == 0 and i == 1 and self.regions[cid].country_id == self.regions[node].country_id:
                nodes.extend([cid])
                connections = connections + 1 + \
                    self._get_connections(cid, nodes)
            cid += 1
        return connections

    def _calculate_scores(self, country_id=-1):
        if country_id == -1:
            self.scores = list(map(lambda x: 0, range(self.countries)))
        for region in self.regions:
            if country_id >= 0 and region.country_id != country_id:
                continue
            score = self._get_connections(region.id, [])
            if score > self.scores[region.country_id]:
                self.scores[region.country_id] = score

    def _plant_population(self):
        for country_id in range(self.countries):
            pool = len(self.country_regions[country_id]
                       ) * INITIAL_POPULATION_PER_REGION
            while pool > 0:
                for region_id in self.country_regions[country_id]:
                    region = self.regions[region_id]
                    if region.population == 0:
                        region.population = 1
                        pool -= 1
                        continue
                    if pool > 0 and region.population < MAX_POPULATION and random.random() > 0.5:
                        pool -= 1
                        region.population += 1

    def _build_country_map(self):
        for region in self.regions:
            self.country_regions[region.country_id].extend([region.id])

    def print_country(self, country_id):
        pops = []
        for i in range(len(self.regions)):
            if self.regions[i].country_id == self.active_player:
                pops.append(self.regions[i].population)
        print('CO: ', country_id, ' Score: ',
              self.scores[country_id], ' Pops: ', pops, ' / ', sum(pops))

    def get_country_sizes(self):
        rets = []
        for i in range(self.countries):
            rets.append(len(self.country_regions[i]))
        return rets

    def load_from_file(self, file_name):
        self.country_regions = []
        self.country_scores = []
        self.regions = []
        self.adjacency_matrix = []
        self.active_player = 0
        self.selected_region = -1

        with open(file_name) as f:
            data = json.load(f)
        self.adjacency_matrix = data['full']

        self.G = nx.Graph()
        self.G.add_nodes_from(range(0, len(self.adjacency_matrix)))

        region_id = 0
        max_country = 0
        for adjacency_vector in self.adjacency_matrix:
            region = Region()
            region.id = region_id
            region.country_id = data['country_map'][str(region_id)] - 1
            region.adjacency_vector = adjacency_vector

            cid = 0
            for connected in adjacency_vector:
                if connected:
                    self.G.add_edge(region_id, cid)
                cid += 1

            self.regions.extend([region])
            if region.country_id > max_country:
                max_country = region.country_id
            region_id += 1

        self.countries = max_country + 1
        for country_id in range(0, self.countries):
            self.country_regions.extend([[]])
            self.country_scores.extend([0])

        self.regions_count = len(self.regions)


if __name__ == '__main__':
    arena = Arena()
    print('SelfTest', os.getcwd())
    arena.load_from_file(os.getcwd().replace('/env', '') + '/models/map.json')
    print('Regions: ', len(arena.regions), ' / Countrzises: ', arena.countries)
    arena.new_game()
    print('Reward: ', arena.act(29, 30))
