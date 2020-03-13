import numpy as np


class GreedyShuffled:
    def __init__(self, n_x, n_y, n_a, data, delta, eps):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.data = data
        self.probabilities = None
        self.name = 'Greedy'
        self.label = 'G'
        self.delta = delta
        self.eps = eps

    def learn(self):
        histories = self.data['h']
        covariates = self.data['x']

        patient_statistics = np.zeros(((2,) * self.n_x + (self.n_y + 1,) * self.n_a + (self.n_a,) + (self.n_y,)), dtype=int)
        for i in range(len(covariates)):
            history = histories[i]
            covariate = covariates[i]
            treatment, outcome = history[-1]
            history = history[:-1]
            index = np.hstack((covariate, np.ones(self.n_a, dtype=int) * -1))
            new = np.zeros((self.n_a, self.n_y), dtype=int)
            for h in history:
                index[h[0] + 1] = h[1]
            new[treatment, outcome] = 1
            ind = tuple(index)
            patient_statistics[ind] += new
        self.probabilities = patient_statistics

        return patient_statistics

    def evaluate(self, patient):
        best_outcome = 0
        x = patient[1]
        y_fac = patient[2]
        y = np.array([-1] * self.n_a)
        stop = False
        history = []
        while not stop:
            state = np.array([x, y])
            prob_matrix = self.probabilities[tuple(np.hstack(state))]
            tot = np.sum(prob_matrix, axis=1)
            tot[tot == 0] = 1
            prob_vec = np.zeros(self.n_a)
            for i in range(best_outcome+1+self.eps, self.n_y):
                prob_vec += prob_matrix[:, i]
            prob_better_vec = np.divide(prob_vec, tot)
            prob_of_finding_better = np.max(prob_better_vec)
            if prob_of_finding_better > self.delta:
                ev_vec = np.zeros(self.n_a)
                for i in range(best_outcome+1+self.eps, self.n_y):
                    ev_vec += prob_matrix[:, i] * i
                ev_vec = np.divide(ev_vec, tot)
                new_treatment = np.argmax(ev_vec)
                outcome = int(y_fac[new_treatment])
                if outcome > best_outcome:
                    best_outcome = outcome
                y[new_treatment] = outcome
                history.append([new_treatment, outcome])
            else:
                stop = True
        return history
