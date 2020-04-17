import numpy as np


class ProbabilityApproximator:
    def __init__(self, n_x, n_a, n_y, data):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.max_possible_outcome = self.n_y - 1
        self.stop_action = self.n_a
        self.data = data

    def calculate_probability(self, probability_of_outcome_approximation, outcome):
        pass

    def to_index(self, state):
        return tuple(np.hstack(state))

    def calculate_probability_greedy(self, prob_matrix, best_outcome, use_expected_value=True):
        tot = np.sum(prob_matrix, axis=1)
        tot[tot == 0] = 1
        ev_vec = np.zeros(self.n_a)
        for i in range(best_outcome + 1, self.n_y):
            if use_expected_value:
                ev_vec += prob_matrix[:, i] * i
            else:
                ev_vec += prob_matrix[:, i]
        ev_vec = np.divide(ev_vec, tot)
        return ev_vec

    def calculate_probability_constraint(self, x, outcomes_state, accuracy):
        pass

