import numpy as np
import matplotlib.pyplot as plt
from DataGenerator.data_generator import split_patients, generate_data
from Database.antibioticsdatabase import AntibioticsDatabase
from Main.SingleEvaluations import AntibioticsDeltaSweepSettings

def plot_sweep_delta(values, times, settings, plot_var=False, split_plot=True):
    plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
    plot_markers = ['s', 'v', 'P', '1', '2', '3', '4']
    plot_lines = ['-', '--', ':', '-.', '-', '--', ':']

    # Extract settings
    load_settings = settings.load_settings
    setup_algorithms = settings.setup_algorithms
    starting_seed, n_data_sets, n_deltas, file_name_prefix = load_settings()
    dist = AntibioticsDatabase(AntibioticsDeltaSweepSettings.n_x, 50, seed=10342)
    training_data, test_data = dist.get_data()
    training_data = split_patients(training_data)
    algs = setup_algorithms(training_data, dist, 0)
    n_algorithms = len(algs)
    deltas = np.linspace(0.0, 1.0, n_deltas)

    values_mean = np.sum(values, 0) / n_data_sets
    times_mean = np.sum(times, 0) / n_data_sets
    values_var = np.zeros((n_deltas, n_algorithms))
    times_var = np.zeros((n_deltas, n_algorithms))
    for i_delta in range(n_deltas):
        for i_alg in range(n_algorithms):
            v_var = 0
            t_var = 0
            for i_data_set in range(n_data_sets):
                v_var += (values_mean[i_delta][i_alg] - values[i_data_set][i_delta][i_alg])**2
                t_var += (times_mean[i_delta][i_alg] - times[i_data_set][i_delta][i_alg])**2
            values_var[i_delta][i_alg] = v_var / (n_data_sets - 1)
            times_var[i_delta][i_alg] = t_var / (n_data_sets - 1)

    # Plot mean treatment effect vs delta
    if not split_plot:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax2 = ax1.twinx()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    ax1.set_title(r'Mean treatment effect/mean search time vs $\delta$')
    ax1.set_xlabel(r'$\delta$')
    ax2.set_xlabel(r'$\delta$')
    ax1.set_ylabel('Efficacy')
    ax2.set_ylabel('Mean search time')
    lns = []
    for i_alg in range(n_algorithms):
        ln1 = ax1.plot(deltas, values_mean[:, i_alg], plot_colors[i_alg] + plot_markers[i_alg] + plot_lines[0],
                       label='{} {}'.format(algs[i_alg].label, 'effect'), markevery=3)
        ln2 = ax2.plot(deltas, times_mean[:, i_alg], plot_colors[i_alg] + plot_markers[i_alg] + plot_lines[1],
                       label='{} {}'.format(algs[i_alg].label, 'time'), markevery=3)
        lns.append(ln1)
        lns.append(ln2)
        if plot_var:
            ln1v = ax1.fill_between(deltas, values_mean[:, i_alg] - values_var[:, i_alg], values_mean[:, i_alg] + values_var[:, i_alg],
                                    facecolor=plot_colors[i_alg], alpha=0.3)
            ln2v = ax2.fill_between(deltas, times_mean[:, i_alg] - times_var[:, i_alg], times_mean[:, i_alg] + times_var[:, i_alg],
                                    facecolor=plot_colors[i_alg], alpha=0.3)
            lns.append(ln1v)
            lns.append(ln2v)
    ax1.grid(True)
    ax2.grid(True)
    plt.rcParams["font.family"] = "serif"
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')
    ax2.legend(lines2, labels2, loc='lower left')
    plt.savefig("saved_values/" + file_name_prefix + "_plot2.png")


if __name__ == '__main__':

    settings = AntibioticsDeltaSweepSettings
    starting_seed, n_data_sets, n_deltas, file_name_prefix = settings.load_settings()
    values = np.load('saved_values/' + file_name_prefix + "values.npy")
    times = np.load('saved_values/' + file_name_prefix + "times.npy")

    plot_sweep_delta(values, times, settings, plot_var=False, split_plot=False)
