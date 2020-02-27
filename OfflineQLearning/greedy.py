from DataGenerator.data_generator import *
import numpy as np


class Greedy:
    def __init__(self, n_x, n_y, n_a):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.probabilities = np.zeros((n_a+1, n_a+1))

    def find_probabilities(self, data):
        h = data['h']

        treatments_effective = np.zeros((self.n_a+1, self.n_a+1))
        treatments_total = np.zeros((self.n_a+1, self.n_a+1))
        #treatments_effective = [0]*n_actions
        #treatments_total = [0]*n_actions

        for history in h:
            intervention = history.pop(-1)
            action, effectiveness = intervention
            index = self._find_index_in_history(history)
            print(index)
            treatments_effective[index] += effectiveness
            treatments_total[index] += 1

        no_history_found = (treatments_total < 1).astype(int)
        treatments_total += no_history_found  # to avoid division by zero

        self.probabilities = np.divide(treatments_effective, treatments_total)

    def _find_index_in_history(self, history):
        index = [-1]*(self.n_a-1)
        for i, h in enumerate(history):
            index[i] = h[0]
        return tuple(index)

n_actions = 3
greedyAlgorithm = Greedy(1, 2, n_actions)
data = generate_data(FredrikDistribution(), 500)
data = split_patients(data)
greedyAlgorithm.find_probabilities(data)
print(greedyAlgorithm.probabilities)

