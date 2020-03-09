from DataGenerator.data_generator import *


class QLearner:
    def __init__(self, n_x, n_y, n_a, learning_rate=0.01, discount_factor=1):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = None

    def to_index(self, state):
        return tuple(np.hstack(state))

    def learn(self, history):
        # Q-table indexed with x, y_0, y_1, y_2, y_3 and a
        self.q_table = np.zeros((2,) * self.n_x + (self.n_y + 1,) * self.n_a + (self.n_a + 1,))

        # Initialize all final states with the rewards for picking that state
        for x in range(len(self.q_table)):
            for y, _ in np.ndenumerate(self.q_table[x]):
                y_t = [-1 if e == self.n_y else e for e in y[0:self.n_a]]
                self.q_table[self.to_index([x, y_t, -1])] = max(y_t)
                if self.q_table[self.to_index([x, y_t, -1])] < 1:
                    self.q_table[self.to_index([x, y_t, -1])] = -np.infty

        for k in range(20000):
            state, action, reward, next_state = history[np.random.randint(0, len(history))]

            self.q_table[self.to_index(state) + (action,)] = self.q_table[self.to_index(state) + (action,)] \
                                                 + self.learning_rate * (reward + self.discount_factor
                                                 * max(self.q_table[self.to_index(next_state)])
                                                 - self.q_table[self.to_index(state) + (action,)])

        return self.q_table

    def evaluate(self, subject):
        if self.q_table is None:
            print("Run learn first!")
            return
        x, y_fac = subject
        y = np.array([-1] * self.n_a)
        history = []
        state = np.array([x, y])
        action = np.argmax(self.q_table[self.to_index(state)])
        while action != self.n_a:
            y[action] = y_fac[action]
            history.append([action, y[action]])
            state = np.array([x, y])
            action = np.argmax(self.q_table[self.to_index(state)])
        return history


def convert_to_sars(data, n_actions):
    x = data['x']
    h = data['h']
    all_sars = []
    for i, patient in enumerate(x):
        actions = np.array([-1] * n_actions)
        for treatment in h[i]:
            action, outcome = treatment
            actions[action] = outcome

        for j in range(len(h[i])):
            temp_actions = actions.copy()
            new_action = h[i][j][0]
            temp_actions[new_action] = -1
            s = np.array([patient, temp_actions])
            a = new_action
            r = -0.1
            new_actions = temp_actions.copy()
            new_actions[new_action] = h[i][j][1]
            s_prime = np.array([patient, new_actions])
            sars = (s, a, r, s_prime)
            all_sars.append(sars)
    return all_sars


n_x = 1
n_z = 2
n_a = 4
n_y = 3
ql = QLearner(n_x, n_y, n_a, learning_rate=0.02)
for i in range(50):
    dist = DiscreteDistribution(n_z, n_x, n_a, n_y, seed=0)
    data = generate_data(dist, 5000)
    data = split_patients(data)
    data = convert_to_sars(data, n_a)
    q = ql.learn(data)
    test_data = generate_test_data(dist, 100)
    for j in range(100):
        print(ql.evaluate(test_data[j]))
    print(q[0, -1, -1, -1, -1])
    print(q[1, -1, -1, -1, -1])
