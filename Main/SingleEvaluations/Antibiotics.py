from multiprocessing.pool import Pool
from Main.SingleEvaluations import AntibioticsSettings
from Main.SingleEvaluations.PlotAntibiotics import *
from Database.antibioticsdatabase import AntibioticsDatabase
import time
import numpy as np
from Algorithms.doctor import Doctor


def get_settings():
    return AntibioticsSettings


def do_work(i_data_set, n_algorithms):
    settings = get_settings()
    starting_seed, n_data_sets, delta, file_name_prefix = settings.load_settings()
    # Setup
    seeds = [x for x in range(starting_seed, starting_seed + n_data_sets)]
    print("Starting set {}".format(i_data_set))
    dist, training_data, test_data = settings.setup_data_sets(seeds[i_data_set])
    n_test_samples = len(test_data)
    n_x = dist.n_x
    n_a = dist.n_a
    n_y = dist.n_y
    res = []
    algorithms = settings.setup_algorithms(training_data, n_x, n_a, n_y, delta)
    for alg in algorithms:
        if type(alg) == Doctor:
            alg.set_data(dist.doctor_data)
    n_algorithms = len(algorithms)
    for alg in algorithms:
        # Train the algorithms
        start = time.time()
        print("\tTraining %s..." % alg.name)
        alg.learn()
        print("\tTraining the %s algorithm took {:.3f} seconds".format(time.time() - start) % alg.name)

        # Evaluate the algorithms
    evaluations = {}
    for alg in algorithms:
        alg_evals = []
        print("Evaluating {}".format(alg.name))
        for i in range(n_test_samples):
            alg_evals.append(alg.evaluate(test_data[i]))
        evaluations[alg.name] = alg_evals

    # Calculate max mean treatment effect over population
    max_mean_treatment_effects = np.zeros((n_algorithms, n_a))
    mean_times = np.zeros(n_algorithms)
    for i_alg, alg in enumerate(algorithms):
        for i_sample in range(n_test_samples):
            treatments = evaluations[alg.name][i_sample]
            best_found = 0
            for i_treatment in range(len(max_mean_treatment_effects[i_alg])):
                if i_treatment < len(treatments):
                    effect = treatments[i_treatment][1]
                    if effect > best_found:
                        best_found = effect
                max_mean_treatment_effects[i_alg][i_treatment] += best_found
            mean_times[i_alg] += len(treatments)

    res.append(max_mean_treatment_effects / n_test_samples)
    res.append(mean_times / n_test_samples)

    return res


if __name__ == '__main__':
    settings = get_settings()

    # Settings
    plot_var = False
    starting_seed, n_data_sets, delta, file_name_prefix = settings.load_settings()

    # Quick hack to get n_algorithms
    tmp_dist = AntibioticsDatabase(AntibioticsSettings.n_x, 50, seed=90821)
    training_data, test_data = tmp_dist.get_data()
    training_data = training_data
    n_x = tmp_dist.n_x
    n_a = tmp_dist.n_a
    n_y = tmp_dist.n_y
    algs = settings.setup_algorithms(training_data, n_x, n_a, n_y, delta)
    n_algorithms = len(algs)

    values = np.zeros((n_data_sets, n_algorithms, n_a))
    times = np.zeros((n_data_sets, n_algorithms))

    main_start = time.time()
    pool = Pool(processes=n_data_sets)
    results = []
    for i in range(n_data_sets):
        results.append(pool.apply(do_work, (i, n_algorithms)))
    for i in range(n_data_sets):
        r = results[i]
        values[i] = r[0]
        times[i] = r[1]

    np.save("saved_values/" + file_name_prefix + "values", values)
    plot_data(values, times,  settings, plot_var)
    print("Total time for delta sweep was {:.3f} seconds".format(time.time() - main_start))