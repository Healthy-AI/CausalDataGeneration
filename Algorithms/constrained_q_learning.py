from Algorithms.q_learning import *
from Algorithms.betterTreatmentConstraint import *
from Algorithms.help_functions import *
import random


class ConstrainedQlearner(QLearner):
    def __init__(self, n_x, n_a, n_y, data, delta=0, epsilon=0):
        constraint = Constraint(data, n_a, n_y-1, delta, epsilon)
        self.better_treatment_constraint = constraint.better_treatment_constraint
        super().__init__(n_x, n_a, n_y, data)
        self.statistics = self.get_patient_statistics()
        # Q-table indexed with x, y_0, y_1, y_2, y_3 and a
        self.q_table = np.zeros((2,) * self.n_x + (self.n_y + 1,) * self.n_a + (self.n_a + 1,))
        self.q_table_done = self.q_table.copy().astype(int)
        self.delta = delta
        self.epsilon = epsilon
        self.name = 'Constrained Q-learning'
        self.label = 'CQL'

    def learn(self):
        possible_histories = list(itertools.product(range(-1, self.n_y), repeat=self.n_a))
        for x in range(len(self.q_table)):
            for history in possible_histories:
                for action in range(self.n_a+1):
                    q = self.q_function(history, action, x)
                    index = self.to_index([x, history]) + (action,)
                    self.q_table[index] = q
                    self.q_table_done[index] = 1

    def q_function(self, history, action, x):
        # if action is stop action, calculate the reward
        if action == self.stop_action:
            index = self.to_index([x, history]) + (action,)
            if self.q_table_done[index] == 1:
                reward = self.q_table[index]
            else:
                reward = self.get_reward(self.stop_action, history, x)
                self.q_table[index] = reward
                self.q_table_done[index] = 1
            return reward
        # if action is already used, set reward to -inf
        if history[action] != -1:
            index = self.to_index([x, history]) + (action,)
            reward = -np.infty
            self.q_table[index] = reward
            self.q_table_done[index] = 1
            return reward
        # else, calculate the sum of the reward for each outcome times its probability
        future_reward = 0
        stat_index = tuple([x]) + tuple(history) + tuple([action])
        tot = np.sum(self.statistics[stat_index], axis=None)
        no_history_found = (tot == 0).astype(int)
        divider = tot + no_history_found
        for outcome in range(self.n_y):
            stat_ind = tuple([x]) + tuple(history) + tuple([action]) + tuple([outcome])
            prob = self.statistics[stat_ind] / divider
            if prob > 0:
                future_qs = []
                future_history = list(history)
                future_history[action] = outcome
                # Among allowed actions, find the one with the greatest reward
                for new_action in range(self.n_a+1):
                    index = self.to_index([x, future_history]) + (new_action,)
                    if self.q_table_done[index] == 1:
                        future_q = self.q_table[index]
                    else:
                        future_q = self.q_function(tuple(future_history), new_action, x)
                        self.q_table[index] = future_q
                        self.q_table_done[index] = 1
                    future_qs.append(future_q)
                max_future_q = np.max(future_qs)
            else:
                max_future_q = 0
            # For each outcome, add the probability times maximal future Q
            future_reward = np.add(future_reward, np.multiply(prob, max_future_q))
        r = self.get_reward(action, history, x)
        q = r + future_reward
        return q

    def get_allowed_actions(self, history):
        allowed_actions = []
        for i, entry in enumerate(history[:self.n_a]):
            if entry == -1:
                allowed_actions.append(i)
        allowed_actions.append(self.stop_action)
        return allowed_actions

    def get_patient_statistics(self):
        histories = self.data['h']
        x_s = self.data['x']

        patient_statistics = np.zeros((2,) * self.n_x + (self.n_y+1,) * self.n_a + (self.n_a,) + (self.n_y,))
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

    def get_reward(self, action, history, x):
        gamma = self.better_treatment_constraint(history, x)
        if action == self.stop_action and gamma == 0:
            return -np.infty
        elif action == self.stop_action and gamma == 1:
            return 0
        elif self.stop_action > action >= 0:
            return -1
        else:
            import sys
            print(gamma, action, history)
            sys.exit()






