import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics


class Distribution:
    def __init__(self, seed=None):
        self.random = np.random.RandomState()
        self.random.seed(seed)

    def draw_z(self):
        pass

    def draw_x(self, z):
        pass

    def draw_a(self, h, x, z):
        pass

    def draw_y(self, a, h, x, z):
        pass


# Creates a random distribution with binary discrete covariates and moderators
class DiscreteDistribution(Distribution):
    def __init__(self, n_z, n_x, n_a, steps_y, outcome_sensitivity_x_z=1, seed=None):
        Distribution.__init__(self, seed)
        self.name = 'Discrete'
        self.n_z = n_z
        self.n_x = n_x
        self.n_a = n_a
        self.steps_y = steps_y
        self.x_weight = outcome_sensitivity_x_z*self.n_z/self.n_x

        self.Pz = np.array(self.random.random(self.n_z))
        self.Px = np.array(self.random.random((self.n_x, self.n_z)))
        self.Px = (self.Px.T / np.sum(self.Px, 1)).T
        self.Pa = np.array(self.random.normal(0, 1, (1 + self.n_x + self.n_a, self.n_a)))
        self.Py = np.array(self.random.normal(0, 1, (self.n_a, 1 + self.n_x + self.n_z, self.steps_y)))
        for coeffs in self.Py:
            coeffs[1:self.n_x+1] *= self.x_weight

    # Draw each Z_i according to the probabilities in Pz
    def draw_z(self):
        z = np.zeros(self.n_z)
        for i in range(self.n_z):
            z[i] = self.random.binomial(1, p=self.Pz[i])
        return z

    def draw_x(self, z):
        weights_z = self.Px * z
        weights_x = np.maximum(np.minimum(np.sum(weights_z, 1), 0.98), 0.02)
        x = np.zeros(self.n_x)
        for i in range(self.n_x):
            x[i] = self.random.binomial(1, p=weights_x[i])
        return x.astype(int)

    def draw_a(self, h, x, z):
        tried_a = np.zeros(self.n_a)
        probs = np.zeros(self.n_a)
        for treatment in h:
            tried_a[treatment[0]] = treatment[1]
        v = np.concatenate(([1], x, tried_a))
        for i in range(self.n_a):
            probs[i] = np.exp(self.Pa.T.dot(v))[i]
        # Decrease the likelihood of testing "similar" treatments,
        # defined as the L2 norm of their x-dependant feature vector
        for treatment in h:
            for a in range(self.n_a):
                probs[a] *= self.calc_a_closeness(treatment[0], a, x)
        den = np.sum(probs)
        probs = probs / den
        return self.random.choice(self.n_a, p=probs)

    def calc_a_closeness(self, a0, a1, x):
        cc = np.exp(self.Py[a0][0:self.n_x+1].T.dot(np.concatenate(([1], x)))) / sum(np.exp(self.Py[a0][0:self.n_x+1].T.dot(np.concatenate(([1], x)))))
        dd = np.exp(self.Py[a1][0:self.n_x+1].T.dot(np.concatenate(([1], x)))) / sum(np.exp(self.Py[a1][0:self.n_x+1].T.dot(np.concatenate(([1], x)))))
        return np.linalg.norm(cc - dd)

    def draw_y(self, a, h, x, z):
        v = np.concatenate(([1], x, z))
        probs = np.zeros(self.steps_y)
        for i in range(self.steps_y):
            probs[i] = np.exp(self.Py[a].T.dot(v)).T[i]
        den = np.sum(probs)
        probs = probs / den
        y = self.random.choice(self.steps_y, p=probs)
        done = False
        if y >= self.steps_y - 1 and self.random.random() < 0.9:
            done = True
        return y, done


class DiscreteDistributionWithStaticOutcomes(DiscreteDistribution):
    def __init__(self, n_z, n_x, n_a, steps_y, outcome_sensitivity_x_z=1, seed=None):
        super().__init__(n_z, n_x, n_a, steps_y, outcome_sensitivity_x_z, seed)
        self.name = "Discrete with static outcomes"

    def draw_y(self, a, h, x, z):
        v = np.concatenate(([1], x, z))
        probs = np.zeros(self.steps_y)
        for i in range(self.steps_y):
            probs[i] = np.exp(self.Py[a].T.dot(v)).T[i]
        den = np.sum(probs)
        probs = probs / den
        y = np.argmax(probs)
        done = False
        if y >= self.steps_y - 1 and self.random.random() < 0.9:
            done = True
        return y, done


class DiscreteDistributionWithSmoothOutcomes(DiscreteDistribution):
    def __init__(self, n_z, n_x, n_a, steps_y, outcome_sensitivity_x_z=1, seed=None):
        super().__init__(n_z, n_x, n_a, steps_y, outcome_sensitivity_x_z, seed)
        self.name = "Discrete with smooth outcomes"

        self.dists = dict()

        self.Py1 = np.array(self.random.normal(0, 1, (self.n_a, 1 + self.n_x + self.n_z)))
        self.Py2 = np.array(self.random.normal(0, 1, (self.n_a, 1 + self.n_x + self.n_z)))
        self.Py2[:, 0] = np.abs(self.Py2[:, 0])
        for coeffs in self.Py1:
            coeffs[1:self.n_x+1] *= self.x_weight
        for coeffs in self.Py2:
            coeffs[1:self.n_x+1] *= self.x_weight
        self.neg_Py1 = ((self.Py1 < 0)*self.Py1).sum(1)
        self.pos_Py1 = ((self.Py1 > 0)*self.Py1).sum(1)
        self.neg_Py2 = ((self.Py2 < 0)*self.Py2).sum(1)
        self.pos_Py2 = ((self.Py2 > 0)*self.Py2).sum(1)

    def calc_a_closeness(self, a0, a1, x):
        d1 = self.Py1[a0] - self.Py1[a1]
        d2 = self.Py2[a0] - self.Py2[a1]

        return np.sum(d1**2) + np.sum(d2**2)

    def draw_y(self, a, h, x, z):
        v = np.concatenate(([1], x, z))
        y0 = self.Py1[a].dot(v)
        y0 = (self.steps_y - 1) * (y0 - self.neg_Py1[a]) / (self.pos_Py1[a] - self.neg_Py1[a])
        gamma = self.Py2[a].dot(v)
        gamma = (gamma - self.neg_Py2[a]) / (self.pos_Py2[a] - self.neg_Py2[a])
        probs = np.zeros(self.steps_y)
        if tuple(v) in self.dists:
            dist = self.dists[tuple(v)]
        else:
            dist = scipy.stats.cauchy(y0, gamma)
            self.dists[tuple(v)] = dist
        for i in range(self.steps_y):
            probs[i] = dist.pdf(i)
        probs = probs / sum(probs)
        y = self.random.choice(self.steps_y, p=probs)
        done = False
        if y >= self.steps_y - 1 and self.random.random() < 0.9:
            done = True
        return y, done


class FredrikDistribution(Distribution):
    def __init__(self):
        super().__init__()
        self.name = 'Fredrik'
    n_a = 3
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
    def __init__(self, seed=None):
        super().__init__(seed)
        self.name = 'New'
    n_a = 3
    pz = np.array([0.3, 0.16, 0.35, 0.13, 0.04, 0.01, 0.01])
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
            for u in used_a:
                ev[u] = 0

        return self.random.choice(3, p=ev/sum(ev))

    def draw_y(self, a, h, x, z):
        y = self.results_array[a][z]
        max_y = 0
        if len(h) > 0:
            max_y = np.max(h, 0)[1]
        if ((max_y == 2 or y == 2) and self.random.random() < 0.90) or self.random.random() < 0.1:
            return y, True
        return y, False
