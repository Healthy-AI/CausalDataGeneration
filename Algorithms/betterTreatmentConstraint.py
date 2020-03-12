import numpy as np
from Algorithms.help_functions import *
import itertools


class Constraint:
    def __init__(self, data, n_actions, max_possible_outcome, delta=0, epsilon=0):
        self.data = data
        self.n_actions = n_actions
        self.histories_to_compare = self.history_to_compare_dict(self.data['h'])
        self.better_treatment_constraint_dict = {}
        self.delta = delta
        self.epsilon = epsilon
        self.max_possible_outcome = max_possible_outcome

    def better_treatment_constraint(self, history):
        try:
            gamma = self.better_treatment_constraint_dict[hash_state(history)]
        except KeyError:
            maxoutcome = max(history)
            if maxoutcome == self.max_possible_outcome:
                return 1
            try:
                similar_patients = self.histories_to_compare[hash_state(history)]
            except KeyError:
                similar_patients = []
            treatments_better = np.zeros(self.n_actions, dtype=int)
            treatments_worse = np.zeros(self.n_actions, dtype=int)
            for patient in similar_patients:
                for intervention in patient:
                    treatment, outcome = intervention
                    if outcome > maxoutcome + self.epsilon:
                        treatments_better[treatment] += 1
                    else:
                        treatments_worse[treatment] += 1
            total = treatments_better + treatments_worse
            no_data_found = (total == 0).astype(int)
            total += no_data_found
            tot = treatments_better / total
            tot_delta_limit = (tot > self.delta).astype(int)
            gamma = 1-max(tot_delta_limit)
            self.better_treatment_constraint_dict[hash_state(history)] = gamma
        return gamma

    def history_to_compare_dict(self, histories):
        state_dict = {}
        for history in histories:
            temp_histories = list(itertools.combinations(history, len(history)-1))
            for temp_history in temp_histories:
                history_hash = hash_history(temp_history, self.n_actions)
                try:
                    state_dict[history_hash].append(history)
                except KeyError:
                    state_dict[history_hash] = [history]
        return state_dict
