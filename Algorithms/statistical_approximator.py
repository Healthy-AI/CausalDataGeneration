from Algorithms.probability_approximator import ProbabilityApproximator
import numpy as np


class StatisticalApproximator(ProbabilityApproximator):
    def __init__(self, n_x, n_a, n_y, data, prior_power=2):
        super().__init__(n_x, n_a, n_y, data)
        self.statistics = self.get_patient_statistics()
        self.prior_power = prior_power
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