import numpy as np
import matplotlib.pyplot as plt

from DataGenerator.data_generator import split_patients
from Database.antibioticsdatabase import AntibioticsDatabase
from Main.SingleEvaluations.Settings import AntibioticsDeltaSweepSettings

def get_settings():
    return AntibioticsDeltaSweepSettings

def plot_time_vs_effect(values, times, settings):
    plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
    plot_markers = ['s', 'v', 'P', '1', '2', '3', '4']
    plot_lines = ['-', '--', ':', '-.']

    setup_algorithms = settings.setup_algorithms
    starting_seed, n_data_sets, n_deltas, file_name_prefix = settings.load_settings()
    dist = AntibioticsDatabase(AntibioticsDeltaSweepSettings.n_x, 50, seed=10342)
    training_data, test_data = dist.get_data()
    training_data = split_patients(training_data)
    algs = setup_algorithms(training_data, dist, 0)
    n_algorithms = len(algs)

    values_mean = np.sum(values, 0) / n_data_sets
    times_mean = np.sum(times, 0) / n_data_sets

    zipped_mean = np.zeros((n_algorithms, 2, n_deltas))
    for i_alg in range(n_algorithms):
        zipped_mean[i_alg][0] = times_mean[:, i_alg]
        zipped_mean[i_alg][1] = values_mean[:, i_alg]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    plt.rcParams["font.family"] = "serif"
    for i_alg in range(n_algorithms):
        ax1.plot(zipped_mean[i_alg, 0], zipped_mean[i_alg, 1], plot_colors[i_alg] + plot_markers[i_alg] + plot_lines[0],
                       label='{}'.format(algs[i_alg].label))
    ax1.invert_xaxis()
    ax1.legend()
    plt.xlabel("Mean time")
    plt.ylabel("Efficacy")
    ax1.grid(True)
    plt.savefig("saved_values/" + file_name_prefix + "_time_vs_effect4.pdf")


if __name__ == '__main__':
    settings = GetSettings()
    starting_seed, n_data_sets, n_deltas, file_name_prefix = settings.load_settings()
    values = np.load('saved_values/' + file_name_prefix + "values.npy")
    times = np.load('saved_values/' + file_name_prefix + "times.npy")

    NPD_file_name_prefix = "AntibioticsNDP"
    NDPvalues = np.load('saved_values/' + NPD_file_name_prefix + "values.npy")
    NDPtimes = np.load('saved_values/' + NPD_file_name_prefix + "times.npy")
    values[:,:,-1] = np.squeeze(NDPvalues)
    times[:,:,-1] = np.squeeze(NDPtimes)

    plot_time_vs_effect(values, times, settings)
