from DataGenerator.data_generator import *
import numpy as np


class Greedy:
    def __init__(self, n_x, n_y, n_a):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.probabilities = np.zeros((n_a+1, 2, n_a+1, 2, n_a))

    def find_probabilities(self, data):
        h = data['h']
        size = (self.n_a+1, 2, self.n_a+1, 2, self.n_a)
        treatments_effective = np.zeros(size)
        treatments_total = np.zeros(size)

        for history in h:
            intervention = history.pop(-1)
            action, effectiveness = intervention
            index = self._find_index_in_history(history, action)
            treatments_effective[index] += effectiveness
            treatments_total[index] += 1

        no_history_found = (treatments_total < 1).astype(int)
        treatments_total += no_history_found  # to avoid division by zero

        self.probabilities = np.divide(treatments_effective, treatments_total)

    def _find_index_in_history(self, history, action):
        index = [-1]*(self.n_a+2)
        for i, h in enumerate(history):
            index[2*i] = h[0]
            index[2*i+1] = h[1]
        index[-1] = action
        return tuple(index)
    # First index is first treatment, second index is outcome of first treatment
    # Third index is second treatment, fourth index is outcome of second treatment
    # E.g. probabilities[0, 0, 1, 0, 2] means that we tried treatment 0 and 1 and got zero on both
    # and are now intervening with treatment 2, returned is predicted outcome of intervention.


n_actions = 3
greedyAlgorithm = Greedy(1, 2, n_actions)
data = generate_data(FredrikDistribution(), 1000)
data = split_patients(data)
greedyAlgorithm.find_probabilities(data)
#print(greedyAlgorithm.probabilities[2, 0, 1, 0, 0])
#print(greedyAlgorithm.probabilities[0, 0, 1, 0, 2])