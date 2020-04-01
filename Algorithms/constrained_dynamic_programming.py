from Algorithms.q_learning import *
from Algorithms.betterTreatmentConstraint import *
from Algorithms.help_functions import *
import random


class ConstrainedDynamicProgramming(QLearner):
    def __init__(self, n_x, n_a, n_y, data, constraint):
        self.better_treatment_constraint = constraint.no_better_treatment_exist
        super().__init__(n_x, n_a, n_y, data)
        self.statistics = self.get_patient_statistics()
        # Q-table indexed with x, y_0, y_1, y_2, y_3 and a
        self.q_table = np.zeros((2,) * self.n_x + (self.n_y + 1,) * self.n_a + (self.n_a + 1,))
        self.q_table_done = self.q_table.copy().astype(bool)
        self.name = 'Constrained Dynamic Programming'
        self.label = 'CDP'

    def learn(self):
        possible_x = list(itertools.product(range(0, 2), repeat=self.n_x))
        possible_histories = list(itertools.product(range(-1, self.n_y), repeat=self.n_a))
        for x in possible_x:
            for history in possible_histories:
                for action in range(self.n_a+1):
                    if not self.q_table_done[self.to_index([x, history, action])]:
                        self.populate_q_value(history, action, x)

    def populate_q_value(self, history, action, x):
        index = self.to_index([x, history]) + (action,)
        future_reward = 0
        # if action is stop action, calculate the reward
        if action == self.stop_action:
            reward = self.get_reward(self.stop_action, history, x)
        # if action is already used, set reward to -inf
        elif history[action] != -1:
            reward = -np.infty
        # else, calculate the sum of the reward for each outcome times its probability
        else:
            p_index = self.to_index([x, [-1]*self.n_a]) + (action,)
            prior_samples = self.statistics[p_index]
            num_prior_samples = np.sum(prior_samples, axis=None)
            p_power = 2 # TODO move to self!
            reward = self.get_reward(action, history, x)
            number_of_samples = np.sum(self.statistics[index], axis=None)
            for outcome in range(self.n_y):
                stats_index = index + tuple([outcome])
                prior = prior_samples[outcome] / (num_prior_samples + 1)
                probability_of_outcome = (self.statistics[stats_index] + prior * p_power**2) / (number_of_samples + p_power**2)
                if probability_of_outcome > 0:
                    future_history = list(history)
                    future_history[action] = outcome
                    # Find the action with the greatest reward
                    for new_action in range(self.n_a+1):
                        if not self.q_table_done[self.to_index([x, future_history]) + (new_action, )]:
                            self.populate_q_value(tuple(future_history), new_action, x)
                    max_future_q = np.max(self.q_table[self.to_index([x, future_history])])
                else:
                    max_future_q = 0
                future_reward = np.add(future_reward, np.multiply(probability_of_outcome, max_future_q))
        self.q_table[index] = reward + future_reward
        self.q_table_done[index] = True

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






