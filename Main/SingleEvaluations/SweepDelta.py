import matplotlib.pyplot as plt
import numpy as np
import time
from Algorithms.constrained_dynamic_programming import generate_data, split_patients
from DataGenerator.distributions import DiscreteDistributionWithSmoothOutcomes
from Main.SingleEvaluations import DeltaSweepSettings

if __name__ == '__main__':
    settings = DeltaSweepSettings
    load_settings = settings.load_settings
    setup_algorithms = settings.setup_algorithms
    setup_data_sets = settings.setup_data_sets

    # Settings
    plot_var = False
    starting_seed, n_data_sets, n_deltas, n_z, n_x, n_a, n_y, n_training_samples, n_test_samples, file_name_prefix = load_settings()

    # Setup
    seeds = [x for x in range(starting_seed, starting_seed + n_data_sets)]
    plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
    plot_markers = ['s', 'v', 'P', '1', '2', '3', '4']
    plot_lines = ['-', '--', ':', '-.']
    deltas = np.linspace(0.0, 1.0, n_deltas)

    # Quick hack to get n_algorithms
    tmp_dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y)
    algs = setup_algorithms(split_patients(generate_data(tmp_dist, 10)), tmp_dist, 0.1)
    n_algorithms = len(algs)

    values = np.zeros((n_data_sets, n_deltas, n_algorithms))
    times = np.zeros((n_data_sets, n_deltas, n_algorithms))

    main_start = time.time()
    for i_data_set in range(n_data_sets):
        print("Starting set {}".format(i_data_set))
        dist, training_data, test_data = setup_data_sets(n_z, n_x, n_a, n_y, n_training_samples, n_test_samples, seeds[i_data_set])
        for i_delta in range(n_deltas):
            print("Evaluating delta = {:.3f}".format(deltas[i_delta]))
            algorithms = setup_algorithms(training_data, dist, deltas[i_delta])
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
                values[i_data_set][i_delta][i_alg] = n_at_max / n_test_samples
                times[i_data_set][i_delta][i_alg] = total_time / n_test_samples
    np.save("saved_values/" + file_name_prefix + "values", values)
    np.save("saved_values/" + file_name_prefix + "times", times)
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
    fig, ax1 = plt.subplots(figsize=(10, 7))
    plt.title('Mean treatment effect/mean search time vs delta')
    plt.xlabel('delta')
    ax2 = ax1.twinx()
    ax1.set_ylabel('Mean treatment effect')
    ax2.set_ylabel('Mean search time')
    lns = []
    for i_alg in range(n_algorithms):
        ln1 = ax1.plot(deltas, values_mean[:, i_alg], plot_colors[i_alg],
                       label='{} {}'.format(algs[i_alg].label, 'effect'))
        ln2 = ax2.plot(deltas, times_mean[:, i_alg], plot_colors[i_alg] + plot_lines[1],
                       label='{} {}'.format(algs[i_alg].label, 'time'))
        lns.append(ln1)
        lns.append(ln2)
        if plot_var:
            ln1v = ax1.fill_between(n_deltas, values_mean[:, i_alg] - values_var[:, i_alg], values_mean[:, i_alg] + values_var[:, i_alg],
                                    facecolor=plot_colors[i_alg], alpha=0.3)
            ln2v = ax2.fill_between(n_deltas, times_mean[:, i_alg] - times_var[:, i_alg], times_mean[:, i_alg] + times_var[:, i_alg],
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