import numpy as np

class NaiveGreedy:
    def __init__(self, n_x, n_a, n_y, data):
        self.n_x = n_x
        self.n_a = n_a
        self.n_y = n_y
        self.max_outcome = self.n_y - 1
        self.data = data
        self.max_outcome_statistics = None

        self.name = 'Naive Greedy'
        self.label = 'NG'

    def learn(self):
        self.max_outcome_statistics = np.zeros((2,) * self.n_x + (self.n_a,) + (2,))
        histories = self.data['h']
        covariates = self.data['x']
        for i, history in enumerate(histories):
            covariate = covariates[i]
            treatment0 = history[0]
            if treatment0[1] == self.max_outcome:
                self.max_outcome_statistics[tuple(np.hstack((covariate, treatment0[0], 0)))] += 1
            else:
                self.max_outcome_statistics[tuple(np.hstack((covariate, treatment0[0], 1)))] += 1

    def evaluate(self, patient):
        z, x, y_fac = patient
        base_statistics = self.max_outcome_statistics[tuple(np.hstack(x))]
        base_probabilities = base_statistics[:, 0] / (base_statistics[:, 0] + base_statistics[:, 1])
        best_outcome = 0
        history = []
        while best_outcome < self.max_outcome and np.max(base_probabilities) > 0:
            a = np.argmax(base_probabilities)
            history.append([a, y_fac[a]])
            if y_fac[a] > best_outcome:
                best_outcome = y_fac[a]
            base_probabilities[a] = 0
        return history
