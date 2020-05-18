import numpy as np


class NaiveGreedy:
    def __init__(self, n_x, n_a, n_y, approximator, max_steps, name='Naive Greedy', label='NG'):
        self.n_x = n_x
        self.n_a = n_a
        self.n_y = n_y
        self.max_outcome = self.n_y - 1
        self.approximator = approximator
        self.max_steps = max_steps

        self.name = name
        self.label = label

    def learn(self):
        pass

    def evaluate(self, patient):
        z, x, y_fac = patient
        h = []
        y = np.array((-1,)*self.n_a)
        best_outcome = 0
        while best_outcome < self.max_outcome and len(h) < self.max_steps:
            available_actions = np.argwhere(y == -1).flatten()
            mask_unknown_actions = get_mask(y_fac)
            base_probabilities = np.array((-np.inf,)*self.n_a)
            for a in available_actions:
                base_probabilities[a] = self.approximator.calculate_probability(x, y, a, self.max_outcome)
            base_probabilities += mask_unknown_actions
            if np.max(base_probabilities) == -np.inf:
                break
            a = np.argmax(base_probabilities)
            y[a] = y_fac[a]
            h.append([a, y_fac[a]])
            if y_fac[a] > best_outcome:
                best_outcome = y_fac[a]
        return h


def get_mask(y_fac):
    mask_unknown_actions = y_fac.copy().astype(float)
    mask_unknown_actions[mask_unknown_actions != -1] = 0
    mask_unknown_actions[mask_unknown_actions == -1] = -np.inf
    return mask_unknown_actions

