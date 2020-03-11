import numpy as np


class GreedyShuffled:
    def __init__(self, n_x, n_y, n_a):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.probabilities = None

    def find_probabilities(self, data):
        histories = data['h']
        dim = []
        for i in range(self.n_a):
            dim.append(self.n_y + 1)
        dim.append(self.n_a)
        dim.append(self.n_y)

        patient_statistics = np.zeros(dim, dtype=int)
        for history in histories:
            treatment, outcome = history[-1]
            history = history[:-1]
            index = np.ones(self.n_a, dtype=int) * -1
            new = np.zeros((self.n_a, self.n_y), dtype=int)
            for h in history:
                index[h[0]] = h[1]
            new[treatment, outcome] = 1
            ind = tuple(index)
            patient_statistics[ind] += new
        self.probabilities = patient_statistics

        return patient_statistics

    def evaluate(self, patient, delta, eps):
        best_outcome = 0
        covariates = patient[1]
        counterfactual_outcomes = patient[2]
        tested_treatments = [-1]*self.n_a
        stop = False
        history = []
        while not stop:
            prob_matrix = self.probabilities[tuple(tested_treatments)]
            tot = np.sum(prob_matrix, axis=1)
            tot[tot == 0] = 1
            prob_vec = np.zeros(self.n_a)
            ev_vec = np.zeros(self.n_a)
            for i in range(best_outcome+1+eps, self.n_y):
                prob_vec += prob_matrix[:, i]
            prob_better_vec = np.divide(prob_vec, tot)
            prob_of_finding_better = np.max(prob_better_vec)
            if prob_of_finding_better > delta:
                for i in range(0, self.n_y):
                    ev_vec += prob_matrix[:, i]
                ev_vec = np.divide(ev_vec, tot)
                new_treatment = np.argmax(ev_vec)
                outcome = int(counterfactual_outcomes[new_treatment])
                if outcome > best_outcome:
                    best_outcome = outcome
                tested_treatments[new_treatment] = outcome
                history.append([new_treatment, outcome])
            else:
                stop = True
        return history
