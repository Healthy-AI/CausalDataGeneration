from DataGenerator.data_generator import *


class QLearner:
    def __init__(self, n_x, n_y, n_a, learning_rate=0.01, discount_factor=1):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def to_index(self, state):
        return tuple(np.hstack(state))

    def learn(self, history):
        # Q-table indexed with x, y_0, y_1, y_2, y_3 and a
        q_table = np.zeros((self.n_x,) + (self.n_y + 1,) * self.n_a + (self.n_a + 1,))

        # Initialize all final states with the rewards for picking that state
        for x in range(len(q_table)):
            for y, _ in np.ndenumerate(q_table[x]):
                y_t = [-1 if e == self.n_y else e for e in y[0:self.n_a]]
                q_table[self.to_index([x, y_t, -1])] = max(y_t)
                if q_table[self.to_index([x, y_t, -1])] < 1:
                    q_table[self.to_index([x, y_t, -1])] = -np.infty

        for k in range(150000):
            state, action, reward, next_state = history[np.random.randint(0, len(history))]

            q_table[self.to_index(state) + (action,)] = q_table[self.to_index(state) + (action,)] + self.learning_rate \
                                            * (reward + self.discount_factor * max(q_table[self.to_index(next_state)])
                                               - q_table[self.to_index(state) + (action,)])

        return q_table


def convert_to_sars(data, n_actions):
    x = data['x']
    h = data['h']
    all_sars = []
    for i, patient in enumerate(x):
        actions = [-1] * n_actions
        for treatment in h[i]:
            action, outcome = treatment
            actions[action] = outcome

        for j in range(len(h[i])):
            temp_actions = actions.copy()
            new_action = h[i][j][0]
            temp_actions[new_action] = -1
            s = [patient, temp_actions]
            a = new_action
            r = -0.7
            new_actions = temp_actions.copy()
            new_actions[new_action] = h[i][j][1]
            s_prime = [patient, new_actions]
            sars = (s, a, r, s_prime)
            all_sars.append(sars)
    return all_sars


n_actions = 3
counts = np.zeros(3)
ql = QLearner(1, 2, n_actions, learning_rate=0.01)
for i in range(50):
    data = generate_data(NewDistribution(), 1500)
    data = split_patients(data)
    data = convert_to_sars(data, n_actions)
    q = ql.learn(data)
    counts[np.argmax(q[0, -1, -1, -1])] += 1
    print(q[0, -1, -1, -1])
print(counts)
