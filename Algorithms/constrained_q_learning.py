from Algorithms.q_learning import *
import numpy as np


class ConstrainedQlearner(QLearner):
    def __init__(self, n_x, n_a, n_y, data, learning_rate=0.01, discount_factor=1):
        super().__init__(n_x, n_a, n_y, data, learning_rate, discount_factor)
        self.statistics = self.get_patient_statistics()
        # Q-table indexed with x, y_0, y_1, y_2, y_3 and a
        self.q_table = np.zeros((self.n_x,) + (self.n_y + 1,) * self.n_a + (self.n_a + 1,))
        self.q_table_done = self.q_table.copy()

    def learn(self):
        history = self.sars_data
        # Initialize all final states with the rewards for picking that state
        for x in range(len(self.q_table)):
            for y, _ in np.ndenumerate(self.q_table[x]):
                y_t = [-1 if e == self.n_y else e for e in y[0:self.n_a]]
                self.q_table[self.to_index([x, y_t, -1])] = self.get_reward(self.stop_action, y_t_to_history(y_t))

        for x in range(self.n_x):
            for i in range(-1, self.n_y-1):
                for j in range(-1, self.n_y-1):
                    for k in range(-1, self.n_y-1):
                        #res =
                        for action in range(0, self.n_a-1):
                            for y in range(0, self.n_y-1):
                                history = [i, j, k]
                                state = [x, history]
                                reward = self.get_reward(action, [[0, i], [1, j], [2, k]])
                                next_state = state.copy()
                                next_state[action] = y
                                if self.q_table[self.to_index(state) + (action,)] != -np.infty:
                                    h = tuple(state[1])
                                    prob = self.statistics[h][action]/np.sum(self.statistics[h][action])

                                    self.q_table[self.to_index(state) + (action,)] = \
                                        reward + self.statistics[h]/np.sum(self.statistics[h], axis=1) + \
                                        np.max(self.q_table[self.to_index(next_state)], 1)

    def fill_table(self):
        for x in range(len(self.q_table)):
            for y, _ in np.ndenumerate(self.q_table[x]):
                y_t = [-1 if e == self.n_y else e for e in y[0:self.n_a]]
                self.q_table[self.to_index([x, y_t, -1])] = self.get_reward(self.stop_action, y_t_to_history(y_t))
                self.q_table_done[self.to_index([x, y_t, -1])] = 1

            possible_histories = list(itertools.product(range(-1, self.n_a), repeat=self.n_y))
            for history in possible_histories:
                for action in range(self.n_a):
                    q = self.q_function(history, action, x)
                    index = self.to_index([x, history, action])
                    self.q_table[index] = q
                    self.q_table_done[index] = 1

    def q_function(self, history, action, x):
        # if action is stop action, calculate the reward
        if action == self.stop_action:
            fake_history = y_t_to_history(history)
            index = self.to_index([x, history, action])
            if self.q_table_done[index] == 1:
                reward = self.q_table[index]
            else:
                reward = self.get_reward(self.stop_action, fake_history)
                self.q_table[index] = reward
                self.q_table_done[index] = 1
            return reward
        # else, calculate the sum of the reward for each outcome times its probability
        future_reward = 0
        tot = np.sum(self.statistics[tuple(history)][action], axis=None)
        no_history_found = (tot == 0).astype(int)
        tot += no_history_found
        for outcome in range(self.n_y):
            prob = self.statistics[tuple(history)][action][outcome] / tot
            if np.max(self.statistics[tuple(history)][action]) == 0:
                prob = 1/self.n_a
            future_qs = []
            future_history = list(history)
            future_history[action] = outcome
            # Look in history and find which actions there are left
            allowed_actions = []
            for i, entry in enumerate(future_history[:self.n_a]):
                if entry == -1:
                    allowed_actions.append(i)
            allowed_actions.append(self.stop_action)
            # Among allowed actions, find the one with the greatest reward
            for new_action in allowed_actions:
                index = self.to_index([x, future_history, new_action])
                if self.q_table_done[index] == 1:
                    future_q = self.q_table[index]
                else:
                    future_q = self.q_function(future_history, new_action, x)
                    self.q_table[index] = future_q
                    self.q_table_done[index] = 1
                future_qs.append(future_q)
            max_future_q = np.max(future_qs)
            # For each outcome, add the probability times maximal future Q
            future_reward = np.add(future_reward, np.multiply(prob, max_future_q))
        fake_history = y_t_to_history(history)
        q = self.get_reward(action, fake_history) + future_reward
        return q

    def better_treatment_constraint(self, history, delta=1, epsilon=0):
        maxoutcome = 0
        if len(history) > 0:
            maxoutcome = max(h[0] for h in history)
            if maxoutcome == self.max_possible_outcome:
                return 1

        similar_patients = []
        for other_patient_history in self.data['h']:
            if np.array_equal(np.sort(history), np.sort(other_patient_history)): # TODO fix comparison
                t = [[h[0], (h[1] > maxoutcome + epsilon)] for h in other_patient_history]
                similar_patients.append(t)
        treatments_better = np.zeros(self.n_a, dtype=int)
        treatments_worse = np.zeros(self.n_a, dtype=int)
        for patient in similar_patients:
            for intervention in patient:
                if intervention[1] == True:
                    treatments_better[intervention[0]] += 1
                if intervention[1] == False:
                    treatments_worse[intervention[0]] += 1
        total = treatments_better + treatments_worse
        no_data_found = (total == 0).astype(int)
        total += no_data_found
        tot = treatments_better / total
        tot_delta_limit = (tot >= delta).astype(int)
        return max(tot_delta_limit)

    def get_patient_statistics(self):
        histories = self.data['h']
        dim = []
        for i in range(self.n_a):
            dim.append(self.n_y + 1)
        dim.append(self.n_a)
        dim.append(self.n_y)

        patient_statistics = np.zeros(dim, dtype=int)
        for history in histories:
            treatment, outcome = history.pop(-1)
            index = np.ones(self.n_a, dtype=int) * -1
            new = np.zeros((self.n_a, self.n_y), dtype=int)
            for h in history:
                index[h[0]] = h[1]
            new[treatment, outcome] = 1
            ind = tuple(index)
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
        r = self.get_reward(a, history)
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

'''
def get_patient_statistics2(data):
    histories = data['h']
    patient_data = {}
    for history in histories:
        treatment, outcome = history.pop(-1)
        entry = np.zeros((n_actions, n_outcomes), dtype=int)
        entry[treatment, outcome] = 1
        hash_string = hash_history(history)
        try:
            patient_data[hash_string] += entry
        except KeyError:
            patient_data[hash_string] = entry
    return patient_data


def hash_history(history):
    flat_list = [item for intervention in history for item in intervention]
    strings = [str(integer) for integer in flat_list]
    hash_string = "".join(strings)
    return hash_string


def get_patient_statistics3(data):
    histories = data['h']
    dim = []
    for i in range(n_actions):
        dim.append(n_outcomes+1)
    dim.append(n_actions)
    dim.append(n_outcomes)

    patient_statistics = np.zeros(dim, dtype=int)
    for history in histories:
        intervention = history.pop(-1)
        index = np.ones(n_actions+2, dtype=int)*n_outcomes
        for h in history:
            index[h[0]] = h[1]
        index[-2] = intervention[0]
        index[-1] = intervention[1]
        ind = tuple(index)
        patient_statistics[ind] += 1

    return patient_statistics
'''

def y_t_to_history(y_t):
    history = []
    for i, entry in enumerate(y_t):
        if entry != -1:
            history.append([i, entry])
    return history


'''

#counts = np.zeros(3)
cql = ConstrainedQlearner(1, n_outcomes, n_actions, learning_rate=0.01)
ql = QLearner(1, n_outcomes, n_actions, learning_rate=0.01)

data = generate_data(NewDistribution(), 3000)

data_dict = split_patients(data)
statistics = get_patient_statistics3(data)

data = convert_to_sars(data_dict, n_actions)
cq = cql.learn(data)
q = ql.learn(ordinary_q_learning.convert_to_sars(data_dict, n_actions))

print('cq')
print(cq)
print('---------')

print('q')
print(q)
print('---------')

print(cq[0, -1, -1, -1])
print(cq[0, 1, -1, -1])
print(cq[0, 1, 2, -1])
print(cq[0, 2, -1, -1])
print(cq[0, 1, 1, -1])
print(cq[0, -1, 1, 1])
print('---------')

print(q[0, -1, -1, -1])
print(q[0, 1, -1, -1])
print(q[0, 1, 2, -1])
print(q[0, 2, -1, -1])
print(q[0, 1, 1, -1])
print(q[0, -1, 1, 1])
#counts[np.argmax(q[0, -1, -1, -1])] += 1
#print(q[0, -1, -1, -1])
#print(counts)
'''