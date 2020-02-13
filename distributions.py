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

    def draw_a(self, h, x):
        return False

    def draw_y(self, a, h, x, z):
        return False


class TestDistribution(Distribution):
    n_treatments = 3

    def __init__(self):
        super().__init__()

    def draw_z(self):
        return self.random.randint(0, 3)

    def draw_x(self, z):
        x0 = max(0, 2**z - self.random.randint(-1, 2))
        x1 = self.random.randint(0, 2)
        return [x0, x1]

    def draw_a(self, h, x):
        possible_a = range(0, 3)
        if len(h) > 0:
            used_a = [x[0] for x in h]
        else:
            used_a = []
        draw_a = [x for x in possible_a if x not in used_a]
        self.random.shuffle(draw_a)
        return draw_a[0]

    def draw_y(self, a, h, x, z):
        assert 0 <= a < self.n_treatments
        y = 0
        if a == 0:
            y = self.random.normal(z)
        elif a == 1:
            y = self.random.uniform()*x[1]
        elif a == 2:
            y = self.random.normal(z - 1)
        return y


class SimpleDistribution(Distribution):
    n_treatments = 4

    def draw_z(self):
        return self.random.randint(0, 3)

    def draw_x(self, z):
        x0 = round(min(max(self.random.normal(z/2), 0), 1))
        x1 = self.random.randint(0, 3)
        return [x0, x1]

    def draw_a(self, h, x):
        possible_a = range(0, 4)
        if len(h) > 0:
            used_a = [x[0] for x in h]
        else:
            used_a = []
        draw_a = [x for x in possible_a if x not in used_a]
        self.random.shuffle(draw_a)
        return draw_a[0]

    def draw_y(self, a, h, x, z):
        if a == 0:
            result = self.random.normal(0.8) - x[0]*self.random.uniform(0, 0.45) + x[1]*self.random.uniform(0, 0.45)
        elif a == 1:
            result = self.random.normal(0.6) - x[0] * self.random.uniform(0, 0.45) \
                     + x[1] * self.random.uniform(0, 0.45) + z*self.random.uniform(0, 0.45)
        elif a == 2:
            result = self.random.normal(0.6) + z*self.random.uniform(0, 0.75)
        elif a == 3:
            result = self.random.normal(0.6) + x[0]*self.random.uniform(0, 1.2) \
                     - x[1]*self.random.uniform(0, 0.5) - z*self.random.uniform(0, 0.45)
        else:
            return False
        return round(min(max(result, 0), 2))


def plot_y():
    di = SimpleDistribution()
    a = 3
    freq = np.zeros((3, 2, 3))
    for z in range(3):
        for x0 in range(2):
            for x1 in range(3):
                values = []
                for k in range(300):
                    values.append(di.draw_y(a, None, [x0, x1], z))
                freq[z][x0][x1] = np.mean(values)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlim((0, 2))
        X0, X1 = np.meshgrid([0, 1], [0, 1, 2])
        surf = ax.plot_wireframe(X0, X1, freq[z].T)
        plt.show()