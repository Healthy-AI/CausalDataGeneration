import numpy as np
from Algorithms.betterTreatmentConstraint import Constraint


class GreedyShuffled2:
    def __init__(self, n_x, n_y, n_a):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.probabilities = None
        self.data = None

    def learn(self, data):
        self.data = data
        histories = data['h']
        covariates = data['x']

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

    def evaluate(self, patient, delta, eps):
        constraint = Constraint(self.data, self.n_a, self.n_y-1, delta, eps)
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
            ev_vec = np.zeros(self.n_a)
            for i in range(best_outcome+1+eps, self.n_y):
                ev_vec += prob_matrix[:, i] * i
            ev_vec = np.divide(ev_vec, tot)
            new_treatment = np.argmax(ev_vec)
            outcome = int(y_fac[new_treatment])
            if outcome > best_outcome:
                best_outcome = outcome
            y[new_treatment] = outcome
            history.append([new_treatment, outcome])
            gamma = constraint.better_treatment_constraint(history)
            if gamma == 1:
                stop = True

        return history
