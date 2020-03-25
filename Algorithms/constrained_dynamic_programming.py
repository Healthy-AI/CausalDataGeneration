from Algorithms.q_learning import *
from Algorithms.betterTreatmentConstraint import *
from Algorithms.help_functions import *
import random


class ConstrainedDynamicProgramming(QLearner):
    def __init__(self, n_x, n_a, n_y, data, delta=0, epsilon=0):
        constraint = Constraint(data, n_a, n_y-1, delta, epsilon)
        self.better_treatment_constraint = constraint.no_better_treatment_exist
        super().__init__(n_x, n_a, n_y, data)
        self.statistics = self.get_patient_statistics()
        # Q-table indexed with x, y_0, y_1, y_2, y_3 and a
        self.q_table = np.zeros((2,) * self.n_x + (self.n_y + 1,) * self.n_a + (self.n_a + 1,))
        self.q_table_done = self.q_table.copy().astype(bool)
        self.delta = delta
        self.epsilon = epsilon
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
            reward = self.get_reward(action, history, x)
            number_of_samples = np.sum(self.statistics[index], axis=None)
            if number_of_samples > 0:
                max_future_actions = []
                for outcome in range(self.n_y):
                    stats_index = index + tuple([outcome])
                    probability_of_outcome = self.statistics[stats_index] / number_of_samples
                    if probability_of_outcome > 0:
                        future_history = list(history)
                        future_history[action] = outcome
                        # Find the action with the greatest reward
                        for new_action in range(self.n_a+1):
                            if not self.q_table_done[self.to_index([x, future_history]) + (new_action, )]:
                                self.populate_q_value(tuple(future_history), new_action, x)
                        max_future_q = np.max(self.q_table[self.to_index([x, future_history])])
                    #max_future_action = np.argmax(self.q_table[self.to_index([x, future_history])])
                    #max_future_actions.append(max_future_action)
                    else:
                        max_future_q = 0
                    # For each outcome, add the probability times maximal future Q
                    #if future_reward == -np.inf or max_future_q == -np.inf:
                    #    future_reward = -np.inf
                    #else:
                    future_reward = np.add(future_reward, np.multiply(probability_of_outcome, max_future_q))

                #if all(action == self.stop_action for action in max_future_actions):
                #    #print('found useless action, prev future_reward', future_reward)
                #    future_reward = -1000
            else:
                future_reward = -np.inf
        '''
        action_indicies = []
        if action != self.stop_action:
            for outcome in range(self.n_y):
                future_history = list(history)
                future_history[action] = outcome
                max_action_index = np.argmax(self.q_table[self.to_index([x, future_history])])
                action_indicies.append(max_action_index)
            if all(action == self.stop_action for action in action_indicies):
                print('found useless action, prev future_reward', future_reward)
                future_reward = -np.inf
        '''
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






