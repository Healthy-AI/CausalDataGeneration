import numpy as np
import matplotlib.pyplot as plt
from Main.SingleEvaluations import AntibioticsSettings, AntibioticsSettings3
from Database.antibioticsdatabase import AntibioticsDatabase
from DataGenerator.data_generator import *
plt.rcParams["font.family"] = "serif"

def plot_data(values, times, settings, plot_var=False):

    plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
    plot_markers = ['s', 'v', 'P', '1', '2', '3', '4']
    plot_lines = ['-', '--', ':', '-.'] + [(i, (1, 4, i, i^2)) for i in range(0, 4)]

    # Extract settings
    load_settings = settings.load_settings
    setup_algorithms = settings.setup_algorithms
    starting_seed, n_data_sets, delta, file_name_prefix = load_settings()
    dist = AntibioticsDatabase(AntibioticsSettings.n_x, 50, seed=10342)
    training_data, test_data = dist.get_data()
    training_data = split_patients(training_data)
    n_x = dist.n_x
    n_a = dist.n_a
    n_y = dist.n_y
    algs = setup_algorithms(dist, training_data, n_x, n_a, n_y, delta)

    n_algorithms = len(algs)

    values_mean = np.sum(values, 0) / n_data_sets
    times_mean = np.sum(times, 0) / n_data_sets
    '''
    tmp = algs[-1]
    algs[-1] = algs[-2]
    algs[-2] = tmp

    tmp = values_mean[-1].copy()
    values_mean[-1] = values_mean[-2]
    values_mean[-2] = tmp

    tmp = times_mean[-1].copy()
    times_mean[-1] = times_mean[-2]
    times_mean[-2] = tmp
    '''
    '''
    values_var = np.zeros(n_algorithms)
    times_var = np.zeros(n_algorithms)
    for i_alg in range(n_algorithms):
        v_var = 0
        t_var = 0
        for i_data_set in range(n_data_sets):
            v_var += (values_mean[i_alg] - values[i_data_set][i_alg])**2
            t_var += (times_mean[i_alg] - times[i_data_set][i_alg])**2
        values_var[i_alg] = v_var / (n_data_sets - 1)
        times_var[i_alg] = t_var / (n_data_sets - 1)
    '''

    x = np.arange(0, n_a)
    x_ticks = list(np.arange(1, n_a + 1))
    plt.figure()
    #plt.title(r'Treatment effect. $\delta$: {}'.format(delta))


    average_max_treatment_effect = sum([max(data[-1]) for data in test_data]) / len(test_data)
    mean_lines = np.linspace(0, 1, n_algorithms)
    #algs[-3].label = "NDP_F"
    #algs[-2].label = "Emulated doctor"
    for i_alg in range(n_algorithms):
        if algs[i_alg].name != "Doctor":
            plt.plot(x, values_mean[i_alg],
                     plot_markers[i_alg] + plot_colors[i_alg], linestyle=plot_lines[i_alg % len(plot_lines)], label=algs[i_alg].label)
            #plt.plot(x, values_mean[i_alg], plot_colors[i_alg], linestyle='-',
            #         alpha=0.3)
            # plt.plot(x, mean_treatment_effects[i_plot], plot_markers[i_plot] + plot_colors[i_plot] + plot_lines[1])
            # plt.fill_between(x, mean_treatment_effects[i_plot], max_mean_treatment_effects[i_plot], color=plot_colors[i_plot], alpha=0.1)
            #plt.axvline(times_mean[i_alg] - 1, ymin=mean_lines[i_alg], ymax=mean_lines[i_alg + 1],
            #            color=plot_colors[i_alg])
            plt.axvline(times_mean[i_alg] - 1, ymin=0, ymax=1,
                        color=plot_colors[i_alg], alpha=0.1)
        else:
            plt.plot(0, values_mean[i_alg][0],
                     plot_markers[i_alg] + plot_colors[i_alg], markersize=20, linestyle=plot_lines[i_alg % len(plot_lines)],
                     linewidth=4, label=algs[i_alg].label)

    plt.grid(True)
    plt.xticks(x, x_ticks)
    plt.plot(x, np.ones(len(x)) * average_max_treatment_effect, linestyle=plot_lines[-1], label='MAX_POSS_AVG')
    plt.ylabel('Mean treatment effect', fontsize=14)
    plt.xlabel('Number of tried treatments', fontsize=14)
    plt.legend(loc='lower right')
    plt.savefig("saved_values/" + file_name_prefix + "_plotNewTest.pdf", bbox_inches='tight', pad_inches=0.02)

    # Plot mean number of treatments tried
    plt.figure()
    plt.title('Search time', fontsize=14)
    plt.ylabel('Mean number of treatments tried', fontsize=14)
    plt.xlabel('Policy')
    x_bars = []
    for i_alg, alg in enumerate(algs):
        x_bars.append(alg.name)
    x_bars = [label.replace(" ", '\n') for label in x_bars]
    rects = plt.bar(x_bars, times_mean)
    for rect in rects:
        h = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 0.90 * h, "%f" % h, ha="center", va="bottom")
    plt.show()


if __name__ == '__main__':
    settings = AntibioticsSettings3
    starting_seed, n_data_sets, delta, file_name_prefix = settings.load_settings()
    values = np.load('saved_values/' + file_name_prefix + "values.npy")
    times = np.load('saved_values/' + file_name_prefix + "times.npy")

    plot_data(values, times, settings, False)
