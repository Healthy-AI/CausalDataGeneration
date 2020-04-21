from Algorithms.q_learning import QLearner
import numpy as np
import itertools


class NaiveDynamicProgramming(QLearner):
    def __init__(self, n_x, n_a, n_y, data, approximator, reward=-0.25, name='Naive Dynamic Programming', label='NDP'):
        super().__init__(n_x, n_a, n_y, data)
        self.table_size = (2,) * self.n_x + (self.n_y + 1,) * self.n_a + (self.n_a + 1,)
        self.q_table = np.zeros(self.table_size)
        self.q_table_done = self.q_table.copy().astype(bool)
        self.name = name
        self.label = label
        self.approximator = approximator
        self.reward = reward

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
            reward = self.get_reward(self.stop_action, history)
        # if action is already used, set reward to -inf
        elif history[action] != -1:
            reward = -np.inf
        # else, calculate the sum of the reward for each outcome times its probability
        else:
            reward = self.get_reward(action, history)
            for outcome in range(self.n_y):
                probability_of_outcome = self.approximator.calculate_probability(x, history, action, outcome)
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

    def get_reward(self, action, history):
        if action == self.stop_action:
            return np.max(history)
        elif self.stop_action > action >= 0:
            return self.reward
        Exception("Error, invalid action {} at {}".format(action, history))

    def reset(self):
        self.q_table = np.zeros(self.table_size)
        self.q_table_done = self.q_table.copy().astype(bool)
