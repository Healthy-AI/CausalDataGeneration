from Algorithms.probability_approximator import ProbabilityApproximator
import numpy as np


class TrueApproximator(ProbabilityApproximator):
    def __init__(self, dist):
        super().__init__(0, 0, 0, None)
        self.dist = dist
        self.y_y_dict = dict()
        self.y_h_dict = dict()

    def prepare_calculation(self, x, history, action):
        return x, history, action

    def calculate_probability(self, probability_of_outcome_approximation, outcome):
        x, history, action = probability_of_outcome_approximation
        tried_history = np.argwhere(np.array(history) != -1).flatten()
        h_list = []
        for h in tried_history:
            h_list.append(np.array((h, history[h])))
        h_list = np.array(h_list)
        return self.calculate_y_given_y_pr(x, outcome, action, h_list)

    def calculate_probability_greedy(self, state, best_outcome):
        total_ev = np.zeros(self.dist.n_a)
        x, outcome_state = state
        for y in range(self.dist.n_y):
            if y > best_outcome:
                for a in range(self.dist.n_a):
                    total_ev[a] += self.calculate_probability(self.prepare_calculation(x, outcome_state, a), y)
        return total_ev

    def calculate_probability_constraint(self, x, outcomes_state, accuracy):
        probs = np.zeros(self.dist.n_a)
        for a in range(self.dist.n_a):
            if outcomes_state[a] == -1:
                for y in range(self.dist.n_y):
                    if y > np.max(outcomes_state):
                        probs[a] += self.calculate_probability(self.prepare_calculation(x, outcomes_state, a), y)
        return np.max(probs)

    def calculate_y_given_y_pr(self, x, outcome, action, history):
        idx = np.hstack([x, outcome, action, history.flatten()])
        idx = tuple(idx)
        if idx in self.y_y_dict:
            return self.y_y_dict[idx]

        total_probability = 0
        if len(history) == 0:
            for z, _ in np.ndenumerate(np.zeros((2,) * self.dist.n_z)):
                total_probability += self.calculate_y_given_z_pr(x, z, action, outcome) * self.dist.get_z_probability(z)
        else:
            for z, _ in np.ndenumerate(np.zeros((2,) * self.dist.n_z)):
                total_probability += self.calculate_z_given_history_pr(x, z, history) *\
                                     self.calculate_y_given_z_pr(x, z, action, outcome)
        self.y_y_dict[idx] = total_probability
        return total_probability

    def calculate_z_given_history_pr(self, x, z, history):
        idx = np.hstack([x, z, history.flatten()])
        idx = tuple(idx)
        if idx in self.y_h_dict:
            return self.y_h_dict[idx]
        total_prob = self.dist.get_z_probability(z)
        history = np.copy(history)
        for h in history:
            action, outcome = h
            total_prob *= self.calculate_y_given_z_pr(x, z, action, outcome)
        for i in range(len(history)):
            h = history[i]
            action, outcome = h
            total_prob /= self.calculate_y_given_y_pr(x, outcome, action, history[i+1:])
        self.y_h_dict[idx] = total_prob
        return total_prob

    def calculate_y_given_z_pr(self, x, z, action, outcome):
        return self.dist.calc_y_weights(action, x, z)[outcome]
