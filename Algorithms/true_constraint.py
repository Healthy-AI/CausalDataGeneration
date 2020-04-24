from Algorithms.better_treatment_constraint import Constraint
from Algorithms.true_approximator import TrueApproximator
from DataGenerator.distributions import DiscreteDistributionWithSmoothOutcomes
import numpy as np

class TrueConstraint(Constraint):
    def __init__(self, dist, approximator, delta=0.0, epsilon=0):
        super().__init__(None, dist.n_a, dist.n_y, None, delta, epsilon)
        self.dist = DiscreteDistributionWithSmoothOutcomes()
        self.approximator = TrueApproximator()
        self.probability_matrix = np.ones((2,) * self.dist.n_x + (self.dist.n_y + 1,) * self.dist.n_a + (self.dist.n_a + 1,)) * -1

    def no_better_treatment_exist(self, current_outcomes, x):
        gamma = 0
        prob = self.calculate_total_probability_of_better(x, current_outcomes)
        return prob < self.delta

    def calculate_total_probability_of_better(self, x, current_outcomes):
        state = np.hstack((x, current_outcomes))
        if self.probability_matrix[state] != -1:
            prob = self.probability_matrix[state]
        else:
            possible_actions = np.argwhere(current_outcomes == -1)
            next_step_probs = np.zeros(self.dist.n_a)
            state = np.hstack((x, current_outcomes))
            for a_next in possible_actions:
                next_step_probs[a_next] = self.calculate_probability_of_better(x, current_outcomes, a_next)

            self.probability_matrix[state] = np.max(next_step_probs)
            prob = self.probability_matrix[state]
        return prob

    def calculate_probability_of_better(self, x, current_outcomes, a):
        best_outcome = np.max(current_outcomes)
        p_1step = 0
        for y in range(best_outcome+1, self.dist.n_y):
            p_1step += self.approximator.calculate_probability(x, current_outcomes, a, y)
        for y in range(0, best_outcome):
            future_outcomes = np.copy(current_outcomes)
            future_outcomes[a] = y

        prob = p_1step + (1 - p_1step) * np.max(next_step_probs)
        return prob