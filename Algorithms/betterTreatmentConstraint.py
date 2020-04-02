import numpy as np
from Algorithms.help_functions import *
import random

class Constraint:
    def __init__(self, data, n_actions, steps_y, prior_weight=2, z_value=1.96, delta=0, epsilon=0):
        self.data = data
        self.n_actions = n_actions
        self.histories_to_compare = self.history_to_compare_dict(self.data['h'], self.data['x'])
        self.better_treatment_constraint_dict = {}
        self.accuracy = prior_weight
        self.optimism = z_value
        self.delta = delta
        self.epsilon = epsilon
        self.max_possible_outcome = steps_y - 1
        self.init_similar_patients = {}
        self.n_outcomes = steps_y

    def no_better_treatment_exist(self, outcomes_state, x):
        dict_index = hash_state(x, outcomes_state)
        gamma = 0
        if dict_index in self.better_treatment_constraint_dict.keys():
            gamma = self.better_treatment_constraint_dict[dict_index]
        else:
            max_outcome = max(outcomes_state)
            if max_outcome == self.max_possible_outcome:
                # If we already found max, stop
                gamma = 1
            elif np.count_nonzero(np.array(outcomes_state) == -1) == 0:
                # If we have tried all treatments, stop
                gamma = 1
            else:
                # Count the number of times each outcome has happened for each action
                if dict_index in self.histories_to_compare.keys():
                    similar_patients = self.histories_to_compare[hash_state(x, outcomes_state)]
                else:
                    similar_patients = []
                history_counts = np.zeros((self.n_actions, 2))
                for patient in similar_patients:
                    treatment, outcome = patient[0]
                    if outcome > max_outcome + self.epsilon:
                        history_counts[treatment][0] += 1
                    else:
                        history_counts[treatment][1] += 1

                # Use the full dataset as prior
                prior_index = hash_state(x, [-1]*self.n_actions)
                if prior_index not in self.histories_to_compare.keys():
                    # If there exists no prior, we can't do any calculations
                    self.better_treatment_constraint_dict[dict_index] = gamma
                    return 1
                prior_patients = self.histories_to_compare[prior_index]
                prior_counts = np.zeros((self.n_actions, 2))
                for patient in prior_patients:
                    treatment, outcome = patient[0]
                    if outcome > max_outcome + self.epsilon:
                        prior_counts[treatment][0] += 1
                    else:
                        prior_counts[treatment][1] += 1

                prior = prior_counts[:, 0] / (np.sum(prior_counts, 1) + (np.sum(prior_counts, 1) == 0))
                estimated_probability = (history_counts[:, 0] + self.accuracy**2 * prior) /\
                                    (np.sum(history_counts, 1) + self.accuracy**2)
                tmp_alpha = (history_counts[:, 0] + self.accuracy**2 * prior) / (np.sum(history_counts, 1) + self.accuracy**2)
                estimated_variance = tmp_alpha * (1 - tmp_alpha) / (np.sum(history_counts, 1) + self.accuracy**2 + 1)
                estimated_stddev = np.sqrt(estimated_variance)
                estimated_bounds = self.optimism * estimated_stddev / np.sqrt(np.sum(history_counts, 1) + self.accuracy**2)

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

    def history_to_compare_dict(self, histories, xs):
        state_dict = {}
        for i, history in enumerate(histories):
            x = xs[i]
            temp_history = history[:-1]
            history_hash = hash_history(x, temp_history, self.n_actions)
            try:
                state_dict[history_hash].append(history)
            except KeyError:
                state_dict[history_hash] = [history]
        return state_dict
