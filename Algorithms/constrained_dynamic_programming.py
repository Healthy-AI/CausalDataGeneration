from OldAlgorithms.q_learning import *
from Algorithms.better_treatment_constraint import *


class ConstrainedDynamicProgramming(QLearner):
    def __init__(self, n_x, n_a, n_y, data, constraint, approximator, name='Constrained Dynamic Programming', label='CDP'):
        super().__init__(n_x, n_a, n_y, data)
        self.constraint = constraint
        # Q-table indexed with x, y_0, y_1, y_2, y_3 and a
        self.table_size = (2,) * self.n_x + (self.n_y + 1,) * self.n_a + (self.n_a + 1,)
        self.q_table = np.zeros(self.table_size)
        self.q_table_done = self.q_table.copy().astype(bool)
        self.name = name
        self.label = label
        self.approximator = approximator
        self.calc_prob_of_outcome = self.approximator.calculate_probability

    def learn(self):
        self.reset()
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
            reward = -np.inf
        # else, calculate the sum of the reward for each outcome times its probability
        else:
            reward = self.get_reward(action, history, x)
            for outcome in range(self.n_y):
                probability_of_outcome = self.calc_prob_of_outcome(x, history, action, outcome)
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

    def get_reward(self, action, history, x):
        gamma = self.constraint.no_better_treatment_exist(history, x)
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

    def reset(self):
        self.q_table = np.zeros(self.table_size)
        self.q_table_done = self.q_table.copy().astype(bool)

    def set_constraint(self, constraint):
        self.constraint = constraint.no_better_treatment_exist

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
        print(action)
        while action != self.n_a and len(history) < self.n_a:
            y[action] = y_fac[action]
            history.append([action, y[action]])
            state = np.array([x, y])

            action_candidates = np.argwhere(self.q_table[self.to_index(state)] ==
                                            np.max(self.q_table[self.to_index(state)]+mask_unknown_actions)).flatten()

            to_remove = []
            for i, a in enumerate(action_candidates):
                if a != self.stop_action and y[a] != -1:
                    to_remove.append(i)
            action_candidates = np.delete(action_candidates, to_remove)
            if len(action_candidates) == 1:
                action = action_candidates[0]
            else:
                probs = np.zeros(self.n_a)
                if self.stop_action in action_candidates:
                    action = self.stop_action
                else:
                    for a in action_candidates:
                        for outcome in range(np.max(y)+1,self.n_y):
                            probs[a] += self.approximator.calculate_probability(x, y, a, outcome)
                    action = np.argmax(probs)
        return history


