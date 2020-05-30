import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import LogScale

from DataGenerator.data_generator import split_patients, generate_data
from DataGenerator.distributions import DiscreteDistributionWithSmoothOutcomes
from Main.SingleEvaluations import DeltaSweepSettings_small, TrueApproxSettings, GApproximatorsSettings, \
    CDPApproximatorsSettings, NaiveVsConstrainedSettings, CDPBoundsSettings, GBoundsSettings, GeneralDeltaSweepSettings, \
    GeneralDeltaSweepSettings2, DataAmountSettings3
plt.rcParams["font.family"] = "serif"

def plot_time_vs_effect(values, times, settings):
    plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
    plot_markers = ['s', 'v', 'P', '1', '2', '3', '4']
    plot_lines = ['-', '--', ':', '-.']*2

    starting_seed, n_data_sets, delta, n_data_set_sizes, n_z, n_x, n_a, n_y, n_training_samples_max, n_test_samples, file_name_prefix = settings.load_settings()
    tmp_dist = DiscreteDistributionWithSmoothOutcomes(3, 1, 5, 3)
    algs = settings.setup_algorithms(split_patients(generate_data(tmp_dist, 10)), tmp_dist, 0.1)
    n_algorithms = len(algs)
    algs[-2].label = 'NDP_H'
    values_mean = np.sum(values, 0) / n_data_sets
    times_mean = np.sum(times, 0) / n_data_sets

    zipped_mean = np.zeros((n_algorithms, 2, n_data_set_sizes))
    for i_alg in range(n_algorithms):
        zipped_mean[i_alg][0] = times_mean[:, i_alg]
        zipped_mean[i_alg][1] = values_mean[:, i_alg]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    for i_alg in range(n_algorithms):
        ax1.plot(zipped_mean[i_alg, 0], zipped_mean[i_alg, 1], plot_colors[i_alg] + plot_markers[i_alg] + plot_lines[i_alg],
                       label='{}'.format(algs[i_alg].label), markevery=3)
    ax1.legend()
    plt.xlabel("Mean time")
    plt.ylabel("Efficacy")
    ax1.grid(True)
    plt.savefig("saved_values/" + file_name_prefix + "_time_vs_effectData.pdf")


if __name__ == '__main__':
    settings = DataAmountSettings3
    starting_seed, n_data_sets, delta, n_data_set_sizes, n_z, n_x, n_a, n_y, n_training_samples_max, n_test_samples, file_name_prefix = settings.load_settings()
    values = np.load('saved_values/' + file_name_prefix + "values.npy")
    times = np.load('saved_values/' + file_name_prefix + "times.npy")
    plot_time_vs_effect(values, times, settings)
