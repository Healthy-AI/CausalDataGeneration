from DataGenerator.data_generator import *
from DataGenerator.data_visualizer import *


class OnlineQLearner:
    def __init__(self, n_x, n_a, n_y, distribution, reward=-0.24, learning_time=10000, learning_rate=0.1, discount_factor=1):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.stop_action = self.n_a
        self.distribution = distribution
        self.step_reward = reward
        self.name = "Online Q-learning"
        self.label = "OQL"
        self.learning_time = learning_time
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = None

        # Online stuff
        self.epsilon = 0.99
        self.max_epsilon = self.epsilon
        self.min_epsilon = 0.03
        self.epsilon_decay = 0.999999
        self.current_patient = {'z': None, 'x': None, 'h': []}


    def to_index(self, state):
        return tuple(np.hstack(state))

    def learn(self):
        # Q-table indexed with x, y_0, y_1, y_2, y_3 and a
        self.q_table = np.zeros(((2,) * self.n_x + (self.n_y + 1,) * self.n_a + (self.n_a + 1,)))

        for k in range(self.learning_time):
            state = self.get_new_sample()
            done = False
            while not done:
                action = self.select_action(state)
                reward, next_state = self.observe(state, action)

                if action == self.stop_action:
                    done = True
                    self.q_table[self.to_index(state) + (action,)] = self.q_table[self.to_index(state) + (action,)] \
                                                        + self.learning_rate \
                                                        * (reward - self.q_table[self.to_index(state) + (action,)])
                else:
                    self.q_table[self.to_index(state) + (action,)] = self.q_table[self.to_index(state) + (action,)] \
                                                         + self.learning_rate * (reward + self.discount_factor
                                                         * max(self.q_table[self.to_index(next_state)])
                                                         - self.q_table[self.to_index(state) + (action,)])
                    state = next_state

            self.epsilon = max(self.min_epsilon, self.epsilon - self.max_epsilon/self.learning_time)
            if k % 1000 == 0:
                print("Episode {}".format(k))
        return self.q_table

    def observe(self, state, action):
        next_state = np.array((state[0], state[1].copy()))
        if action == self.stop_action:
            reward = -100000.0
            for pair in self.current_patient['h']:
                if pair[1] > 0:
                    reward = max(reward, pair[1])
        else:
            y, done = self.distribution.draw_y(action, self.current_patient['h'], self.current_patient['x'],
                                               self.current_patient['z'])
            self.current_patient['h'].append([action, y])
            next_state[1][action] = y
            reward = self.step_reward
        return reward, next_state

    def select_action(self, state):
        possible_actions = np.argwhere(state[1] == -1).flatten()
        possible_actions = np.hstack((possible_actions, self.stop_action))
        if np.random.random() < self.epsilon:
            a = np.random.choice(possible_actions)
            return a
        a = self.argmax_possible(self.q_table[self.to_index(state)], possible_actions)
        return a

    def argmax_possible(self, values, possible_actions):
        c_max = min(values)
        action = np.argmin(values)
        for v in range(len(values)):
            if v in possible_actions:
                if values[v] >= c_max:
                    c_max = values[v]
                    action = v
        return action

    def get_new_sample(self):
        self.current_patient['z'] = self.distribution.draw_z()
        self.current_patient['x'] = self.distribution.draw_x(self.current_patient['z'])
        self.current_patient['h'] = []

        return np.array([self.current_patient['x'], np.array([-1] * self.n_a)])

    def evaluate(self, subject):
        if self.q_table is None:
            print("Run learn first!")
            return
        z, x, y_fac = subject
        y = np.array([-1] * self.n_a)
        history = []
        state = np.array([x, y])
        action = np.argmax(self.q_table[self.to_index(state)])
        while action != self.n_a and len(history) < self.n_a:
            y[action] = y_fac[action]
            history.append([action, y[action]])
            state = np.array([x, y])
            action_candidates = np.argwhere(self.q_table[self.to_index(state)] == np.max(self.q_table[self.to_index(state)])).flatten()
            if len(action_candidates) == 1:
                action = action_candidates[0]
            else:
                # TODO implement a better method for choosing when no valid action is available!
                # TODO choose the treatment with highest mean effect for this x?
                print("Choosing action arbitrarily")
                for a in action_candidates:
                    if a == self.stop_action or y[a] == -1:
                        action = a
                        break
        return history