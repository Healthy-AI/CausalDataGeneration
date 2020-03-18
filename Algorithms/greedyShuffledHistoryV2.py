import numpy as np
from Algorithms.betterTreatmentConstraint import Constraint
from Algorithms.help_functions import *


class GreedyShuffled2:
    def __init__(self, n_x, n_a, n_y, data, delta, eps):
        self.n_x = n_x
        self.n_a = n_a
        self.n_y = n_y
        self.data = data
        self.probabilities = None
        self.name = 'Constrained Greedy'
        self.label = 'CG'
        self.delta = delta
        self.eps = eps
        self.constraint = Constraint(self.data, self.n_a, self.n_y - 1, self.delta, self.eps)

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
                index[h[0] + self.n_x] = h[1]
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
            ev_vec = np.zeros(self.n_a)
            for i in range(best_outcome+1+self.eps, self.n_y):
                ev_vec += prob_matrix[:, i] * i

            ev_vec = np.divide(ev_vec, tot)
            hstate = history_to_state(history, self.n_a)
            for i, hs in enumerate(hstate):
                if hs != -1:
                    ev_vec[i] = -np.infty
            new_treatment = np.argmax(ev_vec)
            outcome = int(y_fac[new_treatment])
            if outcome > best_outcome:
                best_outcome = outcome
            y[new_treatment] = outcome
            history.append([new_treatment, outcome])
            gamma = self.constraint.better_treatment_constraint(history_to_state(history, self.n_a), x)
            if gamma == 1:
                stop = True
        return history
