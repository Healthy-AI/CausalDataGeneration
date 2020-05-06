import matplotlib.pyplot as plt
import numpy as np
import time
from Algorithms.constrained_dynamic_programming import generate_data, split_patients
from DataGenerator.distributions import DiscreteDistributionWithSmoothOutcomes
from Main.SingleEvaluations import DeltaSweepSettings, DataAmountSettings

if __name__ == '__main__':
    settings = DataAmountSettings
    load_settings = settings.load_settings
    setup_algorithms = settings.setup_algorithms
    setup_data_sets = settings.setup_data_sets

    # Settings
    plot_var = True
    starting_seed, n_data_sets, delta, n_data_set_sizes, n_z, n_x, n_a, n_y, n_training_samples_max, n_test_samples, file_name_prefix = load_settings()

    # Setup
    seeds = [x for x in range(starting_seed, starting_seed + n_data_sets)]
    plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
    plot_markers = ['s', 'v', 'P', '1', '2', '3', '4']
    plot_lines = ['-', '--', ':', '-.']
    n_training_samples_array = np.geomspace(10, n_training_samples_max, n_data_set_sizes).astype(int)

    # Quick hack to get n_algorithms
    tmp_dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y)
    algs = setup_algorithms(split_patients(generate_data(tmp_dist, 10)), tmp_dist, 0.1)
    n_algorithms = len(algs)

    values = np.zeros((n_data_sets, n_data_set_sizes, n_algorithms))
    times = np.zeros((n_data_sets, n_data_set_sizes, n_algorithms))

    main_start = time.time()
    for i_data_set in range(n_data_sets):
        print("Starting set {}".format(i_data_set))
        dist, unsplit_training_data, test_data = setup_data_sets(n_z, n_x, n_a, n_y, n_training_samples_max, n_test_samples, seeds[i_data_set])
        for i_size in range(n_data_set_sizes):
            print("Evaluating data set size = {}".format(n_training_samples_array[i_size]))
            d_tmp = {'x': np.copy(unsplit_training_data['x'][0:n_training_samples_array[i_size]]),
                     'h': np.copy(unsplit_training_data['h'][0:n_training_samples_array[i_size]]),
                     'z': np.copy(unsplit_training_data['z'][0:n_training_samples_array[i_size]])}
            training_data = split_patients(d_tmp)

            algorithms = setup_algorithms(training_data, dist, delta)git
            for alg in algorithms:
                start = time.time()
                print("Training {}".format(alg.name))
                alg.learn()
                print("Training {} took {:.3f} seconds".format(alg.name, time.time() - start))
            for i_alg in range(n_algorithms):
                print("Evaluating algorithm {}".format(algorithms[i_alg].name))
                total_time = 0
                n_at_max = 0
                for i_sample in range(len(test_data)):
                    interventions = algorithms[i_alg].evaluate(test_data[i_sample])
                    total_time += len(interventions)
                    best_outcome = max([treatment[1] for treatment in interventions])
                    if best_outcome == np.max(test_data[i_sample][2]):
                        n_at_max += 1
                values[i_data_set][i_size][i_alg] = n_at_max / n_test_samples
                times[i_data_set][i_size][i_alg] = total_time / n_test_samples
    np.save("saved_values/" + file_name_prefix + "values", values)
    np.save("saved_values/" + file_name_prefix + "times", times)
    values_mean = np.sum(values, 0) / n_data_sets
    times_mean = np.sum(times, 0) / n_data_sets
    values_var = np.zeros((n_data_set_sizes, n_algorithms))
    times_var = np.zeros((n_data_set_sizes, n_algorithms))
    for i_size in range(n_data_set_sizes):
        for i_alg in range(n_algorithms):
            v_var = 0
            t_var = 0
            for i_data_set in range(n_data_sets):
                v_var += (values_mean[i_size][i_alg] - values[i_data_set][i_size][i_alg]) ** 2
                t_var += (times_mean[i_size][i_alg] - times[i_data_set][i_size][i_alg]) ** 2
            values_var[i_size][i_alg] = v_var / (n_data_sets - 1)
            times_var[i_size][i_alg] = t_var / (n_data_sets - 1)

    # Plot mean treatment effect vs delta
    fig, ax1 = plt.subplots(figsize=(10, 7))
    plt.title('Mean treatment value/Mean search time vs data set size (delta: {})'.format(delta))
    plt.xlabel('Data set size')
    ax2 = ax1.twinx()
    ax1.set_ylabel('Mean treatment value')
    ax2.set_ylabel('Mean search time')
    lns = []
    for i_alg in range(n_algorithms):
        ln1 = ax1.plot(n_training_samples_array, values_mean[:, i_alg], plot_colors[i_alg],
                       label='{} {}'.format(algs[i_alg].label, 'effect'))
        ln2 = ax2.plot(n_training_samples_array, times_mean[:, i_alg], plot_colors[i_alg] + plot_lines[1],
                       label='{} {}'.format(algs[i_alg].label, 'time'))
        lns.append(ln1)
        lns.append(ln2)
        if plot_var:
            ln1v = ax1.fill_between(n_training_samples_array, values_mean[:, i_alg] - values_var[:, i_alg], values_mean[:, i_alg] + values_var[:, i_alg],
                                    facecolor=plot_colors[i_alg], alpha=0.3)
            ln2v = ax2.fill_between(n_training_samples_array, times_mean[:, i_alg] - times_var[:, i_alg], times_mean[:, i_alg] + times_var[:, i_alg],
                                    facecolor=plot_colors[i_alg], alpha=0.3)
            lns.append(ln1v)
            lns.append(ln2v)
    plt.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lgd = plt.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.04, 0), loc='upper left')
    plt.xscale('log')
    plt.savefig("saved_values/" + file_name_prefix + "_plot.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    print("Total time for delta sweep was {:.3f} seconds".format(time.time() - main_start))