import numpy as np
from Algorithms.help_functions import *
import random

history_prior = False

class Constraint:
    def __init__(self, data, n_actions, steps_y, delta=0, epsilon=0):
        self.data = data
        self.n_actions = n_actions
        self.histories_to_compare = self.history_to_compare_dict(self.data['h'], self.data['x'])
        self.better_treatment_constraint_dict = {}
        self.delta = delta
        self.epsilon = epsilon
        self.max_possible_outcome = steps_y - 1
        self.init_similar_patients = {}
        self.n_outcomes = steps_y

    def no_better_treatment_exist(self, outcomes_state, x):
        try:
            gamma = self.better_treatment_constraint_dict[hash_state(x, outcomes_state)]
        except KeyError:
            maxoutcome = max(outcomes_state)
            if maxoutcome == self.max_possible_outcome:
                return 1
            not_tested = np.where(np.array(outcomes_state) == -1)[0]
            tested = np.where(np.array(outcomes_state) != -1)[0]
            if len(not_tested) == 0:
                return 1
            try:
                similar_patients = self.histories_to_compare[hash_state(x, outcomes_state)]
            except KeyError:
                # Checks each "history" that is 1-off
                if history_prior:
                    similar_patients = []
                    for i in range(self.n_actions):
                        if outcomes_state[i] != -1:
                            tmp_state = list(outcomes_state)
                            tmp_state[i] = -1
                            tmp_state = tuple(tmp_state)
                            key = hash_state(x, tmp_state)
                            if key in self.histories_to_compare.keys():
                                similar_patients.extend(self.histories_to_compare[key])
                # Uses the prior from no tried treatments
                else:
                    tmp_state = tuple([-1] * self.n_actions)
                    state_hash = hash_state(x, tmp_state)
                    if state_hash in self.histories_to_compare:
                        similar_patients = self.histories_to_compare[state_hash]
                    else:
                        similar_patients = []
            treatments_better = np.zeros(self.n_actions, dtype=int)
            treatments_worse = np.zeros(self.n_actions, dtype=int)
            for patient in similar_patients:
                intervention = patient[-1]
                treatment, outcome = intervention
                if outcome > maxoutcome + self.epsilon:
                    treatments_better[treatment] += 1
                else:
                    treatments_worse[treatment] += 1
            # Only check the treatments that have not been tested yet
            treatments_better[tested] = 0
            treatments_worse[tested] = 0
            total = treatments_better + treatments_worse
            no_data_found = (total == 0).astype(int)
            total += no_data_found
            probability_of_better = treatments_better / total

            # Sets broad prior

            try:
                treatments_and_outcomes = self.init_similar_patients[hash_x(x)]
            except KeyError:
                init_state = tuple([-1]*len(outcomes_state))
                state_hash = hash_state(x, init_state)
                if state_hash in self.histories_to_compare:
                    similar_patients = self.histories_to_compare[state_hash]
                treatments_and_outcomes = np.zeros((self.n_actions, self.n_outcomes), dtype=int)
                for patient in similar_patients:
                    for intervention in patient:
                        treatment, outcome = intervention
                        treatments_and_outcomes[treatment, outcome] += 1
                self.init_similar_patients[hash_x(x)] = treatments_and_outcomes
            for i in range(len(treatments_better)):
                if no_data_found[i] == 1:
                    total = np.sum(treatments_and_outcomes[i])
                    if total != 0:
                        probability_of_better[i] = np.sum(treatments_and_outcomes[i][max(maxoutcome, 0):])/total
                    else:
                        probability_of_better[i] = 0

            tot_delta_limit = (probability_of_better > self.delta).astype(int)
            gamma = 1-max(tot_delta_limit)
            self.better_treatment_constraint_dict[hash_state(x, outcomes_state)] = gamma
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
