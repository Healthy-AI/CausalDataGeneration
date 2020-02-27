from DataGenerator.data_generator import *


class Greedy:
    def __init__(self, n_x, n_y, n_a):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.probabilities = [1/n_a]*n_a  # init at equal probabilities

    def find_probabilities(self, data):
        h = data['h']

        treatments_effective = [0]*n_actions
        treatments_total = [0]*n_actions

        for history in h:
            for intervention in history:
                action, effectiveness = intervention
                treatments_effective[action] += effectiveness
                treatments_total[action] += 1

        for action in range(n_actions):
            self.probabilities[action] = treatments_effective[action]/treatments_total[action]

        tot = sum(self.probabilities)
        for action in range(n_actions):
            self.probabilities[action] /= tot


n_actions = 3
greedyAlgorithm = Greedy(1, 2, n_actions)
data = generate_data(FredrikDistribution(), 500)
greedyAlgorithm.find_probabilities(data)

