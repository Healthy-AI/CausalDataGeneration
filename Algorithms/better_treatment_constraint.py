import numpy as np
from Algorithms.help_functions import *
import random


class Constraint:
    def __init__(self, data, n_actions, steps_y, approximator, prior_weight=2, z_value=1.96, delta=0, epsilon=0):
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
                best_probability = np.max(estimated_probability)

                if best_probability < self.delta:
                    gamma = 1
            self.better_treatment_constraint_dict[dict_index] = gamma
        return gamma

    def history_to_compare_dict_alt(self, histories, xs):
        state_dict = {}
        for i, history in enumerate(histories):
            x = xs[i]
            for j in range(0, len(history)):
                temp_history = history[:j]
                history_hash = hash_history(x, temp_history, self.n_actions)
                for k in range(1, len(history)-j+1):
                    h = history[:j+k]
                    try:
                        state_dict[history_hash].append(h)
                    except KeyError:
                        state_dict[history_hash] = [h]
        return state_dict


