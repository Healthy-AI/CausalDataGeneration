import matplotlib.pyplot as plt
import numpy as np
import time
from Algorithms.constrained_dynamic_programming import generate_data, split_patients
from DataGenerator.distributions import DiscreteDistributionWithSmoothOutcomes
from Main.SingleEvaluations import DeltaSweepSettings, GApproximatorsSettings, CDPApproximatorsSettings
from Main.SingleEvaluations.PlotSweepDelta import plot_sweep_delta

if __name__ == '__main__':
    settings = CDPApproximatorsSettings
    load_settings = settings.load_settings
    setup_algorithms = settings.setup_algorithms
    setup_data_sets = settings.setup_data_sets

    # Settings
    plot_var = False
    starting_seed, n_data_sets, n_deltas, n_z, n_x, n_a, n_y, n_training_samples, n_test_samples, file_name_prefix = load_settings()

    # Setup
    seeds = [x for x in range(starting_seed, starting_seed + n_data_sets)]
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
    plot_sweep_delta(values, times, settings, plot_var)
    print("Total time for delta sweep was {:.3f} seconds".format(time.time() - main_start))