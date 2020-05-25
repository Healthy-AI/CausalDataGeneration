from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np
import time
from Algorithms.constrained_dynamic_programming import generate_data, split_patients
from Database.antibioticsdatabase import AntibioticsDatabase
from Main.SingleEvaluations import AntibioticsDeltaSweepSettings, NDPAntibioticsDeltaSweepSettings
from Main.SingleEvaluations.PlotTimeVsEffectAntibiotics import plot_time_vs_effect


def get_settings():
    return NDPAntibioticsDeltaSweepSettings

def do_work(i_data_set, n_algorithms):
    settings = get_settings()
    starting_seed, n_data_sets, n_deltas, file_name_prefix = settings.load_settings()
    res = np.zeros((2, n_deltas, n_algorithms))
    # Setup
    seeds = [x for x in range(starting_seed, starting_seed + n_data_sets)]
    deltas = np.linspace(0.0, 1.0, n_deltas)

    print("Starting set {}".format(i_data_set))
    dist, training_data, test_data = settings.setup_data_sets(seeds[i_data_set])
    n_test_samples = len(test_data)
    for i_delta in range(n_deltas):
        print("Evaluating delta = {:.3f}".format(deltas[i_delta]))
        algorithms = settings.setup_algorithms(training_data, dist, deltas[i_delta])
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
            res[0][i_delta][i_alg] = n_at_max / n_test_samples
            res[1][i_delta][i_alg] = total_time / n_test_samples
    return res

if __name__ == '__main__':
    settings = get_settings()

    # Settings
    plot_var = False
    starting_seed, n_data_sets, n_deltas, file_name_prefix = settings.load_settings()

    # Quick hack to get n_algorithms
    dist, training_data, test_data = settings.setup_data_sets(10342)
    n_x = dist.n_x
    n_a = dist.n_a
    n_y = dist.n_y
    algorithms = settings.setup_algorithms(training_data, dist, 0)
    n_algorithms = len(algorithms)

    values = np.zeros((n_data_sets, n_deltas, n_algorithms))
    times = np.zeros((n_data_sets, n_deltas, n_algorithms))

    main_start = time.time()
    pool = Pool(processes=n_data_sets)
    results = []
    for i in range(n_data_sets):
        results.append(pool.apply_async(do_work, (i, n_algorithms)))
    for i in range(n_data_sets):
        r = results[i].get()
        values[i] = r[0]
        times[i] = r[1]

    np.save("saved_values/" + file_name_prefix + "values", values)
    np.save("saved_values/" + file_name_prefix + "times", times)
    plot_time_vs_effect(values, times, settings)
    print("Total time for delta sweep was {:.3f} seconds".format(time.time() - main_start))