import numpy as np


class Distribution():

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


class SimpleDistribution(Distribution):
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

