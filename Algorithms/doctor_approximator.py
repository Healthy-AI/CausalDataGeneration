
import numpy as np
from Algorithms.help_functions import *


class DoctorApproximator:
    def __init__(self, n_x, n_a, n_y, data, epsilon=0):
        self.n_x = n_x
        self.n_a = n_a
        self.n_y = n_y
        self.data = data
        self.statistics = self.get_patient_statistics()

    def get_patient_statistics(self):
        histories = self.data['h']
        x_s = self.data['x']

        patient_statistics = np.zeros((2,) * self.n_x + (self.n_y+1,) * self.n_a + (self.n_a,) + (self.n_y,), dtype=int)
        for i, history in enumerate(histories):
            x = x_s[i]
            treatment, outcome = history[-1]
            chopped_history = history[:-1]
            index = np.ones(self.n_a, dtype=int) * -1
            new = np.zeros((self.n_a, self.n_y), dtype=int)
            for h in chopped_history:
                index[h[0]] = h[1]
            new[treatment, outcome] = 1
            ind = tuple(x) + tuple(index)
            patient_statistics[ind] += new

        return patient_statistics

    def calculate_probability(self, x, state):
        index = tuple(np.hstack((x, state)))
        actions_and_outcomes = self.statistics[index]
        actions = np.array(np.sum(actions_and_outcomes, axis=1), dtype=float)
        return actions
