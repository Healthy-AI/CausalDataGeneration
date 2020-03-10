import numpy as np


class GreedyShuffled:
    def __init__(self, n_x, n_y, n_a):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.size = tuple([n_y]*n_a + [n_a, n_y])
        self.probabilities = np.zeros(self.size)

    def find_probabilities(self, data):
        histories = data['h']
        dim = []
        for i in range(self.n_a):
            dim.append(self.n_y + 1)
        dim.append(self.n_a)
        dim.append(self.n_y)

        patient_statistics = np.zeros(dim, dtype=int)
        for history in histories:
            treatment, outcome = history.pop(-1)
            index = np.ones(self.n_a, dtype=int) * -1
            new = np.zeros((self.n_a, self.n_y), dtype=int)
            for h in history:
                index[h[0]] = h[1]
            new[treatment, outcome] = 1
            ind = tuple(index)
            patient_statistics[ind] += new
        patient_statistics = np.sum(patient_statistics, 3)
        self.probabilities = patient_statistics

        return patient_statistics

    def evaluate(self, patient, delta, eps):
        best_outcome = 0
        tested_treatments = [-1]*self.n_a
        stop = False
        while not stop:
            prob_matrix = self.probabilities[tested_treatments]
            tot = np.sum(prob_matrix, axis=None)
            prob_vec = np.zeros(self.n_a)
            for i in range(best_outcome+1, self.n_y-1):
                prob_vec += prob_matrix[:, i]
            prob_better_vec = prob_vec / tot
            prob_of_finding_better = np.max(prob_better_vec)
            if prob_of_finding_better > delta:
                new_treatment = np.argmax(prob_better_vec)
                outcome = patient[1][new_treatment]
                tested_treatments[new_treatment] = outcome
            else:
                stop = True

        return tested_treatments
