from DataGenerator.data_generator import *
from DataGenerator.data_visualizer import *
from Algorithms.help_functions import *

class QLearner:
    def __init__(self, n_x, n_a, n_y, data, reward=-0.1, learning_time=10000, learning_rate=0.01, discount_factor=1):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.max_possible_outcome = self.n_y - 1
        self.stop_action = self.n_a
        self.data = data
        self.step_reward = reward
        self.learning_time = learning_time
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = None
        self.name = 'Q-learning'
        self.label = 'QL'
        self.statistics = None
        self.sars_data = self.convert_to_sars()

    def to_index(self, state):
        return tuple(np.hstack(state))

    def learn(self):
        # Q-table indexed with x, y_0, y_1, y_2, y_3 and a
        self.q_table = np.zeros((2,) * self.n_x + (self.n_y + 1,) * self.n_a + (self.n_a + 1,))
        modified = self.q_table.copy()
        history = self.sars_data

        # Initialize all final states with the rewards for picking that state
        for x, _ in np.ndenumerate(np.zeros((2,)*self.n_x)):
            for y, _ in np.ndenumerate(self.q_table[x]):
                y_t = [-1 if e == self.n_y else e for e in y[0:self.n_a]]
                self.q_table[self.to_index([x, y_t, -1])] = max(y_t)
                if self.q_table[self.to_index([x, y_t, -1])] < 1:
                    self.q_table[self.to_index([x, y_t, -1])] = -np.infty
                modified[self.to_index(([x, y_t, -1]))] = True

        for k in range(self.learning_time):
            state, action, reward, next_state = history[np.random.randint(0, len(history))]


            self.q_table[self.to_index(state) + (action,)] = self.q_table[self.to_index(state) + (action,)] \
                                                 + self.learning_rate * (reward + self.discount_factor
                                                 * max(self.q_table[self.to_index(next_state)])
                                                 - self.q_table[self.to_index(state) + (action,)])
            modified[self.to_index(state) + (action,)] = True

        # Hack to make all unmodified cells a bad action
        self.q_table += (modified-1) * self.n_a * self.n_y * 10.0
        return self.q_table

    def evaluate(self, subject):
        if self.q_table is None:
            print("Run learn first!")
            return
        z, x, y_fac = subject
        y = np.array([-1] * self.n_a)
        history = []
        state = np.array([x, y])
        mask_unknown_actions = y_fac.copy().astype(float)
        mask_unknown_actions[mask_unknown_actions != -1] = 0
        mask_unknown_actions[mask_unknown_actions == -1] = -np.inf
        mask_unknown_actions = np.append(mask_unknown_actions, 0)
        action = np.argmax(self.q_table[self.to_index(state)]+mask_unknown_actions)

        while action != self.n_a and len(history) < self.n_a:
            y[action] = y_fac[action]
            history.append([action, y[action]])
            state = np.array([x, y])

            action_candidates = np.argwhere(self.q_table[self.to_index(state)] ==
                                            np.max(self.q_table[self.to_index(state)]+mask_unknown_actions)).flatten()
            if len(action_candidates) == 1:
                action = action_candidates[0]
            else:
                to_remove = []
                for i, a in enumerate(action_candidates):
                    if a != self.stop_action and y[a] != -1:
                        to_remove.append(i)
                action_candidates = np.delete(action_candidates, to_remove)

                if False:#self.statistics is not None:
                    highest_expected_outcome = -1
                    action = None
                    for a in action_candidates:
                        index = self.to_index([x, y, action])
                        total_n_samples = np.sum(self.statistics[index])
                        total_n_samples += (total_n_samples == 0).astype(int)
                        probabilities = self.statistics[index]/total_n_samples
                        expected_outcome = np.sum(probabilities*np.arange(0, self.n_y))
                        if expected_outcome > highest_expected_outcome:
                            highest_expected_outcome = expected_outcome
                            action = a
                else:
                    print("Choosing action arbitrarily")
                    for a in action_candidates:
                        try:
                            if y[a] == -1:
                                action = a
                                break
                        except IndexError:
                            action = self.stop_action
                            break
        return history

    def convert_to_sars(self):
        x = self.data['x']
        h = self.data['h']
        all_sars = []
        for i, patient in enumerate(x):
            actions = np.array([-1] * self.n_a)
            for treatment in h[i]:
                action, outcome = treatment
                actions[action] = outcome

            for j in range(len(h[i])):
                temp_actions = actions.copy()
                new_action = h[i][j][0]
                temp_actions[new_action] = -1
                s = np.array([patient, temp_actions])
                a = new_action
                r = self.step_reward
                new_actions = temp_actions.copy()
                new_actions[new_action] = h[i][j][1]
                s_prime = np.array([patient, new_actions])
                sars = (s, a, r, s_prime)
                all_sars.append(sars)
        return all_sars
