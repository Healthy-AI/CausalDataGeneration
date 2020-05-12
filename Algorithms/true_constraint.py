from Algorithms.better_treatment_constraint import Constraint
from Algorithms.exact_approximator import ExactApproximator
from DataGenerator.distributions import DiscreteDistributionWithSmoothOutcomes
import numpy as np

class TrueConstraint(Constraint):
    def __init__(self, dist, approximator, delta=0.0, epsilon=0):
        super().__init__(None, dist.n_a, dist.n_y, None, delta, epsilon)
        self.dist = dist
        self.approximator = approximator
        self.delta = delta
        self.probability_matrix = np.ones((2,) * self.dist.n_x + (self.dist.n_y + 1,) * self.dist.n_a) * -1

    def no_better_treatment_exist(self, current_outcomes, x):
        prob = self.calculate_total_probability_of_better(x, current_outcomes)
        if prob <= self.delta:
            return 1
        return 0

    def calculate_total_probability_of_better(self, x, current_outcomes):
        state = tuple(np.hstack((x, current_outcomes)))
        if np.max(current_outcomes) == -1:
            prob = 1
        elif self.probability_matrix[state] != -1:
            prob = self.probability_matrix[state]
        else:
            possible_actions = np.argwhere(np.array(current_outcomes) == -1).flatten()
            next_step_probs = np.zeros(self.dist.n_a)
            for a_next in possible_actions:
                next_step_probs[a_next] = self.calculate_probability_of_better(x, current_outcomes, a_next)

            self.probability_matrix[state] = np.max(next_step_probs)
            prob = self.probability_matrix[state]
        return prob

    def calculate_probability_of_better(self, x, current_outcomes, a):
        best_outcome = np.max(current_outcomes)
        if best_outcome >= self.dist.n_y - 1:
            return 0
        probs = np.zeros(self.dist.n_y)
        prob = 0
        for y in range(self.dist.n_y):
            probs[y] = self.approximator.calculate_probability(x, current_outcomes, a, y)
        for y in range(self.dist.n_y):
            if y <= best_outcome:
                future_outcomes = np.copy(current_outcomes)
                future_outcomes[a] = y
                prob += probs[y] * self.calculate_total_probability_of_better(x, future_outcomes)
            else:
                prob += probs[y]

        return prob