from Algorithms.probability_approximator import ProbabilityApproximator
import numpy as np
from Algorithms.help_functions import *


class StatisticalApproximator(ProbabilityApproximator):
    def __init__(self, n_x, n_a, n_y, data, epsilon=0, prior_power=2):
        super().__init__(n_x, n_a, n_y, data)
        self.statistics = self.get_patient_statistics()
        self.data = data
        self.histories_to_compare = self.history_to_compare_dict(self.data['h'], self.data['x'])
        self.prior_power = prior_power
        self.epsilon = epsilon
        self.name = 'statistics approximator'

    def prepare_calculation(self, x, history, action):
        index = self.to_index([x, history]) + (action,)
        p_index = self.to_index([x, [-1]*self.n_a]) + (action,)
        prior_samples = self.statistics[p_index]
        num_prior_samples = np.sum(prior_samples, axis=None)
        number_of_samples = np.sum(self.statistics[index], axis=None)
        return index, prior_samples, num_prior_samples, number_of_samples

    def calculate_probability(self, probability_of_outcome_approximation, outcome):
        index, prior_samples, num_prior_samples, number_of_samples = probability_of_outcome_approximation
        stats_index = index + tuple([outcome])
        prior = prior_samples[outcome] / (num_prior_samples + (prior_samples[outcome] == 0))
        probability_of_outcome = (self.statistics[stats_index] + prior * self.prior_power**2) / (number_of_samples + self.prior_power**2)
        return probability_of_outcome

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

    def calculate_probability_constraint(self, x, outcomes_state, accuracy):
        dict_index = hash_state(x, outcomes_state)
        max_outcome = max(outcomes_state)
        # Count the number of times each outcome has happened for each action
        if dict_index in self.histories_to_compare.keys():
            similar_patients = self.histories_to_compare[hash_state(x, outcomes_state)]
        else:
            similar_patients = []
        history_counts = self.find_counts(similar_patients, max_outcome)
        # Use the full dataset as prior
        prior_index = hash_state(x, [-1] * self.n_a)
        if prior_index not in self.histories_to_compare.keys():
            # If there exists no prior, we can't do any calculations
            return 1
        prior_patients = self.histories_to_compare[prior_index]
        prior_counts = self.find_counts(prior_patients, max_outcome)

        prior = prior_counts[:, 0] / (np.sum(prior_counts, 1) + (np.sum(prior_counts, 1) == 0))
        estimated_probability = (history_counts[:, 0] + accuracy ** 2 * prior) / \
                                (np.sum(history_counts, 1) + accuracy ** 2)
        '''
        tmp_alpha = (history_counts[:, 0] + self.accuracy ** 2 * prior) / (np.sum(history_counts, 1) + self.accuracy ** 2)
        estimated_variance = tmp_alpha * (1 - tmp_alpha) / (np.sum(history_counts, 1) + self.accuracy ** 2 + 1)
        estimated_stddev = np.sqrt(estimated_variance)
        estimated_bounds = self.optimism * estimated_stddev / np.sqrt(np.sum(history_counts, 1) + self.accuracy ** 2)
        '''
        return estimated_probability

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
