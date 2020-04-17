from Algorithms.probability_approximator import ProbabilityApproximator
import numpy as np
from Algorithms.help_functions import *


class StatisticalApproximator(ProbabilityApproximator):
    def __init__(self, n_x, n_a, n_y, data, epsilon=0):
        super().__init__(n_x, n_a, n_y, data)
        self.statistics = self.get_patient_statistics()
        self.data = data
        self.histories_to_compare = self.history_to_compare_dict(self.data['h'], self.data['x'])
        self.epsilon = epsilon
        self.type = type
        self.name = 'statistics approximator'

    def calculate_probability(self, x, history, action, outcome):
        probs = self.full_history_prior(x, history, action, kernel=self.kernel_gaussian)
        return probs[outcome]

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

    def calculate_probability_greedy(self, state, best_outcome, use_expected_value=True):
        prob_matrix = self.statistics[tuple(np.hstack(state))]
        return super(StatisticalApproximator, self).calculate_probability_greedy(prob_matrix, best_outcome, use_expected_value)

    def generate_all_possible_histories(self, state):
        to_generate_for = [tuple(state)]
        all_histories = []
        while len(to_generate_for) > 0:
            all_histories.append(to_generate_for[0])
            s = to_generate_for.pop(0)
            for i in range(len(s)):
                if s[i] != -1:
                    s_tmp = np.copy(s)
                    s_tmp[i] = -1
                    s_tmp = tuple(s_tmp)
                    if s_tmp not in to_generate_for:
                        to_generate_for.append(s_tmp)
        return np.copy(all_histories)

    def calc_history_distance(self, history, other_history):
        h_order = len(np.argwhere(history != -1))
        oh_order = len(np.argwhere(other_history != -1))
        diff = np.abs(h_order - oh_order)
        return diff

    def kernel_gaussian(self, current_state, historical_state):
        diff = self.calc_history_distance(current_state, historical_state)
        return np.exp(-diff**2)

    def kernel_laplace(self, current_state, historical_state):
        diff = self.calc_history_distance(current_state, historical_state)
        return np.exp(-diff)

    def get_probabilities(self, x, state, action):
        stats = self.statistics[tuple(np.hstack((x, state, action)))]
        den = np.sum(stats)
        if den == 0:
            den = 1
        return stats / den

    def full_history_prior(self, x, state, action, kernel):
        histories = self.generate_all_possible_histories(state)
        total_probabilities = np.zeros(self.n_y)
        total_kernel = 0
        for h in histories:
            k = kernel(state, h)
            total_probabilities += k * self.get_probabilities(x, h, action)
            total_kernel += k
        return total_probabilities / k

    def calculate_probability_constraint(self, x, state):
        best_outcome = np.max(state)
        full_probabilities = np.zeros((self.n_a, self.n_y))
        for a in range(self.n_a):
            full_probabilities[a] = self.full_history_prior(x, state, a, self.kernel_gaussian)
        better_probabilities = np.zeros(self.n_a)
        for a in range(self.n_a):
            total = 0
            for y in range(self.n_y):
                total += full_probabilities[a][y]
                if y > best_outcome:
                    better_probabilities[a] += full_probabilities[a][y]
            if total == 0:
                total = 1
            better_probabilities[a] /= total
        return better_probabilities

    def history_to_compare_dict(self, histories, xs):
        state_dict = {}
        for i, history in enumerate(histories):
            x = xs[i]
            temp_history = history[:-1]
            history_hash = hash_history(x, temp_history, self.n_a)
            try:
                state_dict[history_hash].append(history)
            except KeyError:
                state_dict[history_hash] = [history]
        return state_dict

    def find_counts(self, patients, max_outcome):
        counts = np.zeros((self.n_a, 2))
        for patient in patients:
            treatment, outcome = patient[-1]
            if outcome > max_outcome + self.epsilon:
                counts[treatment][0] += 1
            else:
                counts[treatment][1] += 1
        return counts
