import numpy as np
from Algorithms.help_functions import *
import random


class Constraint:
    def __init__(self, data, n_actions, steps_y, approximator, prior_weight=2, z_value=1.96, delta=0, epsilon=0, bound='upper'):
        self.data = data
        self.n_actions = n_actions
        self.better_treatment_constraint_dict = {}
        self.accuracy = prior_weight
        self.optimism = z_value
        self.delta = delta
        self.epsilon = epsilon
        self.max_possible_outcome = steps_y - 1
        self.init_similar_patients = {}
        self.n_outcomes = steps_y
        self.approximator = approximator
        if bound == 'upper':
            self.bound = self.upper_bound_constraint
        elif bound == 'lower':
            self.bound = self.lower_bound_constraint

    def no_better_treatment_exist(self, outcomes_state, x):
        dict_index = hash_state(x, outcomes_state)
        gamma = 0
        if dict_index in self.better_treatment_constraint_dict.keys():
            gamma = self.better_treatment_constraint_dict[dict_index]
        else:
            max_outcome = max(outcomes_state)  # double-check input here
            if max_outcome == self.max_possible_outcome:
                # If we already found max, stop
                gamma = 1
            elif np.count_nonzero(np.array(outcomes_state) == -1) == 0:
                # If we have tried all treatments, stop
                gamma = 1
            else:
                estimated_probability = self.approximator.calculate_probability_constraint(x, outcomes_state)
                probability_limit = self.bound(estimated_probability)

                if probability_limit < self.delta:
                    gamma = 1
            self.better_treatment_constraint_dict[dict_index] = gamma
        return gamma

    def upper_bound_constraint(self, estimated_probabilities):
        sum_probability = np.sum(estimated_probabilities)
        return sum_probability

    def lower_bound_constraint(self, estimated_probabilities):
        max_probability = np.max(estimated_probabilities)
        return max_probability