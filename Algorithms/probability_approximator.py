import numpy as np


class ProbabilityApproximator:
    def __init__(self, n_x, n_a, n_y, data):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.max_possible_outcome = self.n_y - 1
        self.stop_action = self.n_a
        self.data = data

    def prepare_calculation(self, x, history, action):
        pass

    def calculate_probability(self, probability_of_outcome_approximation, outcome):
        pass

    def to_index(self, state):
        return tuple(np.hstack(state))

