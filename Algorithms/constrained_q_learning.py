from Algorithms.q_learning import *
import Algorithms.q_learning as ordinary_q_learning
import numpy as np


class ConstrainedQlearner(QLearner):
    def __init__(self, n_x, n_y, n_a, learning_rate=0.01, discount_factor=1):
        super().__init__(n_x, n_y, n_a, learning_rate, discount_factor)

    def learn(self, history):
        # Q-table indexed with x, y_0, y_1, y_2, y_3 and a
        q_table = np.zeros((self.n_x,) + (self.n_y + 1,) * self.n_a + (self.n_a + 1,))

        # Initialize all final states with the rewards for picking that state
        for x in range(len(q_table)):
            for y, _ in np.ndenumerate(q_table[x]):
                y_t = [-1 if e == self.n_y else e for e in y[0:self.n_a]]
                q_table[self.to_index([x, y_t, -1])] = get_reward(stop_action, y_t_to_history(y_t))
        print(q_table)

        for k in range(150000):
            state, action, reward, next_state = history[np.random.randint(0, len(history))]

            if q_table[self.to_index(state) + (action,)] != -np.infty:
                q_table[self.to_index(state) + (action,)] += self.learning_rate * (reward + self.discount_factor *
                    np.max(q_table[self.to_index(next_state)]) - q_table[self.to_index(state) + (action,)])

        return q_table


def get_patient_statistics(data):
    histories = data['h']
    dim = []
    for i in range(n_actions):
        dim.append(2)
    for i in range(2):
        dim.append(n_outcomes)

    patient_statistics = np.zeros(dim, dtype=int)
    for history in histories:
        index = np.zeros(n_actions+2, dtype=int)
        for h in history:
            index[h[0]] = 1
        for intervention in history:
            index[-2] = intervention[0]
            index[-1] = intervention[1]
            ind = tuple(index)
            print(patient_statistics.shape, ind, index)
            print(patient_statistics[ind], ind)
            patient_statistics[ind] += 1

    return patient_statistics


def y_t_to_history(y_t):
    history = []
    for i, entry in enumerate(y_t):
        if entry != -1:
            history.append([i, entry])
    return history


def convert_to_sars(data, n_actions):
    x = data['x']
    h = data['h']
    all_sars = []
    for i, patient in enumerate(x):
        actions = [-1] * n_actions
        history = h[i]
        for treatment in history:
            action, outcome = treatment
            actions[action] = outcome

        for j in range(len(h[i])):
            new_action = h[i][j][0]
            new_outcome = h[i][j][1]
            sars = get_sars(actions, new_action, patient, history, new_outcome)
            all_sars.append(sars)

    return all_sars


def get_sars(actions, new_action, patient, history, outcome):
    temp_actions = actions.copy()

    temp_actions[new_action] = -1
    s = [patient, temp_actions]
    a = new_action
    r = get_reward(a, history)
    new_actions = temp_actions.copy()
    new_actions[new_action] = outcome
    s_prime = [patient, new_actions]
    sars = (s, a, r, s_prime)
    return sars


def get_reward(action, history):
    gamma = better_treatment_constraint(history)
    if action == stop_action and gamma == 0:
        return -np.infty
    elif action == stop_action and gamma == 1:
        return 0
    elif stop_action > action >= 0:
        return -1
    else:
        import sys
        print(gamma, action, history)
        sys.exit()


def better_treatment_constraint(history, delta=1, epsilon=0):
    maxoutcome = 0
    if len(history) > 0:
        maxoutcome = max(h[0] for h in history)
        if maxoutcome == max_possible_outcome:
            return 1

    similar_patients = []
    for other_patient_history in data_dict['h']:
        if sorted(history) in sorted(other_patient_history) or sorted(history) == sorted(other_patient_history):

            t = [[h[0], (h[1] > maxoutcome + epsilon)] for h in other_patient_history]
            similar_patients.append(t)
    treatments_better = np.zeros(n_actions)
    treatments_worse = np.zeros(n_actions)
    for patient in similar_patients:
        for intervention in patient:
            if intervention[1] == True:
                treatments_better[intervention[0]] += 1
            if intervention[1] == False:
                treatments_worse[intervention[0]] += 1
    total = treatments_better + treatments_worse
    no_data_found = (total == 0).astype(int)
    total += no_data_found
    tot = treatments_better/total
    tot_delta_limit = (tot >= delta).astype(int)
    return max(tot_delta_limit)



n_actions = 3
n_outcomes = 3
max_possible_outcome = 2
stop_action = 3
#counts = np.zeros(3)
cql = ConstrainedQlearner(1, n_outcomes, n_actions, learning_rate=0.01)
ql = QLearner(1, n_outcomes, n_actions, learning_rate=0.01)

data = generate_data(NewDistribution(), 3000)
statistics = get_patient_statistics(data)
print(statistics)
print(statistics[0, 0, 0])
print(statistics[1, 1, 1])
print(statistics[1, 1, 0])
print('------')


import sys
sys.exit()
data_dict = split_patients(data)
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
print('---------')

print(q[0, -1, -1, -1])
print(q[0, 1, -1, -1])
print(q[0, 1, 2, -1])
print(q[0, 2, -1, -1])
#counts[np.argmax(q[0, -1, -1, -1])] += 1
#print(q[0, -1, -1, -1])
#print(counts)