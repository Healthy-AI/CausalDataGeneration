import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Distribution:

    def __init__(self, seed=None):
        self.random = np.random.RandomState()
        self.random.seed(seed)

    def draw_z(self):
        return False

    def draw_x(self, z):
        return False

    def draw_a(self, h, x, z):
        return False

    def draw_y(self, a, h, x, z):
        return False


class SimpleDistribution(Distribution):
    n_treatments = 4
    Pzx = np.array([[0.5, 0.35, 0.15], [0.45, 0.25, 0.3]])
    Px = np.array([1 / 3, 2 / 3])
    Pz = np.array([0.467, 0.283, 0.25])

    Pxz = (Pzx.T * Px)
    Pxz = Pxz.T / Pz

    def draw_z(self):
        return int(self.random.choice([0, 1, 2], p=self.Pz))

    def draw_x(self, z):
        return self.random.choice(2, p=self.Pxz.T[z]/sum(self.Pxz.T[z]))

    def draw_a(self, h, x, z):
        possible_a = range(0, 4)
        if len(h) > 0:
            used_a = [x[0] for x in h]
        else:
            used_a = []
        draw_a = [x for x in possible_a if x not in used_a]
        self.random.shuffle(draw_a)
        return draw_a[0]

    def draw_y(self, a, h, x, z):
        ys = [[[0, 0], [0, 1], [1, 2]],
              [[0, 0], [2, 2], [0, 0]],
              [[2, 2], [1, 0], [0, 0]],
              [[1, 1], [1, 1], [1, 1]]]
        return ys[a][z][x]


class SkewedDistribution(SimpleDistribution):

    def __init__(self):
        super().__init__()

    def draw_a(self, h, x, z):
        possible_a = list(range(0, 4))
        if len(h) > 0:
            used_a = [x[0] for x in h]
        else:
            used_a = []
        possible_a = [a for a in possible_a if a not in used_a]
        if x == 1 and 3 not in used_a and self.random.random() < 0.5:
            return 3
        else:
            self.random.shuffle(possible_a)
            return possible_a[0]


class FredrikDistribution(Distribution):
    n_treatments = 3
    treatment_weights = np.array([0.65, 0.6, 0.4])
    results_array = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]])
    z_weights = np.array([0.45, 0.20, 0.20, 0.15])

    def draw_z(self):
        return self.random.choice(4, p=self.z_weights)

    def draw_x(self, z):
        return [0]

    def draw_a(self, h, x, z):
        weights = np.sum(self.z_weights * self.results_array, 1)
        if len(h) > 0:
            treatment_found = np.max(h, 0)[1]
            used_a = [u[0] for u in h]
            if treatment_found:
                weights = np.array([1, 1, 1])
                for u in used_a:
                    weights[u] = 0
            else:
                a_weights = self.results_array.copy()
                for u in used_a:
                    a_weights = a_weights - a_weights[u]
                    a_weights = np.maximum(a_weights, 0)
                a_weights = self.z_weights * a_weights
                weights = np.sum(a_weights, 1)

        return self.random.choice(3, p=weights/sum(weights))

    def draw_y(self, a, h, x, z):
        result = self.results_array[a][z]
        if len(h) > 0:
            best_result = np.max(h, 0)[1]
            if max(result, best_result) == 1 and self.random.random() < 0.85:
                return result, True
        return result, False


class NewDistribution(Distribution):
    n_treatments = 3
    pz = np.array([0.3, 0.15, 0.35, 0.14, 0.04, 0.01, 0.01])
    results_array = np.array([[2, 2, 1, 1, 1, 1, 0],
                              [2, 1, 0, 2, 1, 1, 2],
                              [1, 0, 2, 0, 1, 2, 0]])

    def draw_z(self):
        return self.random.choice(7, p=self.pz)

    def draw_x(self, z):
        return [0]

    def draw_a(self, h, x, z):
        ev = np.sum(self.pz * (self.results_array == 2), 1)
        if len(h) > 0:
            used_a = [u[0] for u in h]
            max_y = np.max(h, 0)[1]
            if max_y == 0:
                ev[0] *= 2
            elif max_y == 1:
                ev[2] *= 2
            elif max_y == 2:
                ev[1] *= 2
            for u in used_a:
                ev[u] = 0

        return self.random.choice(3, p=ev/sum(ev))

    def draw_y(self, a, h, x, z):
        y = self.results_array[a][z]
        max_y = 0
        if len(h) > 0:
            max_y = np.max(h, 0)[1]
        if (max_y == 2 or y == 2) and self.random.random() < 0.85:
            return y, True
        return y, False
