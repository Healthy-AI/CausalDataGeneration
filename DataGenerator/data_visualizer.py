from DataGenerator.data_generator import *
import matplotlib.pyplot as plt

class DataVisualizer:
    def __init__(self, n_z, n_x, n_a, n_y):
        self.n_z = n_z
        self.n_x = n_x
        self.n_a = n_a
        self.n_y = n_y

    def plot_x(self, data):
        counts = np.zeros(((2,)*self.n_x))
        for q in data['x']:
            counts[tuple(q)] += 1

        # Plot stuff
        n = 2**self.n_x
        labels = []
        values = []
        for idx, val in np.ndenumerate(counts):
            labels.append(str(idx))
            values.append(val)
        ind = np.arange(n)
        plt.bar(ind, values)
        plt.ylabel('Samples')
        plt.xlabel('X-values')
        plt.xticks(ind, labels)
        plt.show()

    def print_y(self, data, x):
        counts = np.zeros(((2,)*self.n_z) + (self.n_a,))
        mean = np.zeros(((2,)*self.n_z) + (self.n_a,))
        zeds = np.zeros(((2,)*self.n_z))
        totals = np.zeros(self.n_a)
        treat_counts = np.zeros(self.n_a)
        print("For X = {}".format(str(x)))
        for i in range(len(data['x'])):
            if np.array_equal(data['x'][i], x):
                history = data['h'][i]
                z = data['z'][i].astype(int)
                zeds[tuple(z)] += 1
                for h in history:
                    a = h[0]
                    counts[tuple(z) + (a,)] += 1
                    mean[tuple(z) + (a,)] += h[1]
                    totals[a] += h[1]
                    treat_counts[a] += 1

        mean = np.divide(mean, counts)
        zeds = np.divide(zeds, np.sum(zeds))
        i = 1
        print("{:10} {:4}".format("Z", "P(Z)"))
        for idx, _ in np.ndenumerate(mean):
            z_idx = idx[:-1]
            if idx[-1] % i == 0:
                print("{:10}".format(str(z_idx)), end=' ')
                print("{:.2f}".format(zeds[z_idx]), end=' ')
                for j in range(len(mean[z_idx])):
                    print("{:.2f}".format(mean[z_idx][j]), end=' ')
                print()
            i += 1
        totals = np.divide(totals, treat_counts)
        print("{:10} {:.2f}".format("Total", 1.00), end=' ')
        for j in range(self.n_a):
            print("{:.2f}".format(totals[j]), end=' ')
        print()
