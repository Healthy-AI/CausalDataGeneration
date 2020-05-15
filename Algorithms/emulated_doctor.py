import numpy as np
from Algorithms.help_functions import *


class EmulatedDoctor:
    def __init__(self, n_x, n_a, n_y, data, approximator):
        self.n_x = n_x
        self.n_y = n_y
        self.n_a = n_a
        self.max_outcome = self.n_y - 1
        self.name = "Emulated doctor"
        self.label = "EmDoc"
        self.data = data
        self.approximator = approximator

    def learn(self):
        pass

    def evaluate(self, patient):
        best_outcome = 0
        x = patient[1]
        y_fac = patient[2]
        y = np.array([-1] * self.n_a)
        stop = False
        history = []
        while not stop or len(history) >= self.n_a:
            probabilities = self.approximator.calculate_probability(x, y)
            hstate = history_to_state(history, self.n_a)
            for i, hs in enumerate(hstate):
                if hs != -1:
                    probabilities[i] = -np.inf
            decision_probabilities = probabilities
            new_treatment = np.argmax(decision_probabilities)
            if np.max(decision_probabilities) == -np.inf:
                break
            outcome = int(y_fac[new_treatment])
            if outcome > best_outcome:
                best_outcome = outcome
            y[new_treatment] = outcome
            history.append([new_treatment, outcome])
            if outcome == self.max_outcome:
                stop = True
        return history

