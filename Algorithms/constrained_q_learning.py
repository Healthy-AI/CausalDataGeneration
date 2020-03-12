from Algorithms.q_learning import *
from Algorithms.betterTreatmentConstraint import *
from Algorithms.help_functions import *


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

    def learn(self):
        for x in range(len(self.q_table)):
            possible_histories = list(itertools.product(range(-1, self.n_y), repeat=self.n_a))
            for history in possible_histories:
                for action in range(self.n_a+1):
                    q = self.q_function(history, action, x)
                    index = self.to_index([x, history, action])
                    self.q_table[index] = q
                    self.q_table_done[index] = 1

    def q_function(self, history, action, x):
        # if action is stop action, calculate the reward
        if action == self.stop_action:
            index = self.to_index([x, history, action])
            if self.q_table_done[index] == 1:
                reward = self.q_table[index]
            else:
                reward = self.get_reward(self.stop_action, history)
                self.q_table[index] = reward
                self.q_table_done[index] = 1
            return reward
        if history[action] != -1 and -1 in history:
            index = self.to_index([x, history, action])
            reward = -np.infty
            self.q_table[index] = reward
            self.q_table_done[index] = 1
            return reward
        # else, calculate the sum of the reward for each outcome times its probability
        future_reward = 0
        tot = np.sum(self.statistics[x][tuple(history)][action], axis=None)
        no_history_found = (tot == 0).astype(int)
        tot += no_history_found
        for outcome in range(self.n_y):
            prob = self.statistics[x][tuple(history)][action][outcome] / tot
            future_qs = []
            if prob > 0:
                future_history = list(history)
                future_history[action] = outcome
                # Look in history and find which actions there are left
                #allowed_actions = self.get_allowed_actions(future_history)
                # Among allowed actions, find the one with the greatest reward
                for new_action in range(self.n_a+1):
                    index = self.to_index([x, future_history, new_action])
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
        r = self.get_reward(action, history)
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
        dim = []
        dim.append(self.n_x+1)
        for i in range(self.n_a):
            dim.append(self.n_y + 1)
        dim.append(self.n_a)
        dim.append(self.n_y)

        patient_statistics = np.zeros(dim, dtype=int)
        for i, history in enumerate(histories):
            x = x_s[i]
            treatment, outcome = history[-1]
            history = history[:-1]
            index = np.ones(self.n_a, dtype=int) * -1
            new = np.zeros((self.n_a, self.n_y), dtype=int)
            for h in history:
                index[h[0]] = h[1]
            new[treatment, outcome] = 1
            ind = tuple(x) + tuple(index)
            patient_statistics[ind] += new

        return patient_statistics

    def get_reward(self, action, history):
        gamma = self.better_treatment_constraint(history)
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

    def get_sars(self, actions, new_action, patient, history, outcome):
        temp_actions = actions.copy()

        temp_actions[new_action] = -1
        s = [patient, temp_actions]
        a = new_action
        r = self.get_reward(a, history_to_state(history, self.n_a))
        new_actions = temp_actions.copy()
        new_actions[new_action] = outcome
        s_prime = [patient, new_actions]
        sars = (s, a, r, s_prime)
        return sars

    def convert_to_sars(self):
        x = self.data['x']
        h = self.data['h']
        all_sars = []
        for i, patient in enumerate(x):
            actions = [-1] * self.n_a
            history = h[i]
            for treatment in history:
                action, outcome = treatment
                actions[action] = outcome

            for j in range(len(h[i])):
                new_action = h[i][j][0]
                new_outcome = h[i][j][1]
                sars = self.get_sars(actions, new_action, patient, history, new_outcome)
                all_sars.append(sars)

        return all_sars








