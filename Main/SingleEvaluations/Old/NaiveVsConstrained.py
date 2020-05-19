from Algorithms.naive_dynamic_programming import NaiveDynamicProgramming
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.naive_greedy import NaiveGreedy
from Algorithms.constrained_greedy import ConstrainedGreedy
from Algorithms.exact_approximator import ExactApproximator
from DataGenerator.data_generator import *
import time
from Algorithms.better_treatment_constraint import Constraint
from Algorithms.statistical_approximator import StatisticalApproximator

if __name__ == '__main__':
    # Training values
    #seed = 90821  # Used for both synthetic and real data
    n_tests = 100
    seeds = [np.random.randint(0, 10000) for i in range(n_tests)]
    print(seeds)
    n_z = 3
    n_x = 1
    n_a = 5
    n_y = 3
    training_episodes = 750000
    n_training_samples = 15000
    n_test_samples = 3000
    delta = 0.3
    epsilon = 0
    reward = -0.25
    # for grid search

    # Plot values
    treatment_slack = 0  # Eg, how close to max must we be to be considered "good enough"
    plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
    plot_markers = ['s', 'v', 'P', '1', '2', '3', '4']
    plot_lines = [(i, (1, 4, 1, 4)) for i in range(0, 8)]
    alt_plot_lines = ['-', '--', ':', '-.']

    plot_mean_treatment_effect = False
    plot_treatment_efficiency = False
    plot_delta_efficiency = False
    plot_search_time = False
    plot_strictly_better = False
    plot_delta_grid_search = False
    delta_grid_search_percentage = False
    fixed_scale = False
    plotbools = [plot_mean_treatment_effect, plot_treatment_efficiency, plot_delta_efficiency, plot_search_time,
                 plot_strictly_better]
    main_start = time.time()

    n_algorithms = 5
    mean_n_tests_results = np.zeros((n_algorithms, n_tests))
    efficiency_results = np.zeros((n_algorithms, n_tests))
    for i, seed in enumerate(seeds):
        main_start = time.time()
        print("Starting run {} of {}".format(i, len(seeds)))
        # Generate the data
        dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
        #dist.print_moderator_statistics()
        #dist.print_covariate_statistics()
        #dist.print_treatment_statistics()
        #dist.print_detailed_treatment_statistics()

        print("Generating data")
        training_data = generate_data(dist, n_training_samples)
        test_data = generate_test_data(dist, n_test_samples)

        split_training_data = split_patients(training_data)

        print("Initializing statistical approximator")
        statistical_approximation = StatisticalApproximator(n_x, n_a, n_y, split_training_data, prior_mode='gaussian')
        true_approximation = ExactApproximator(dist)
        # print("Initializing {} took {:.3f} seconds".format(statistical_approximation.name, time.time() - start))

        print("Initializing Constraint")
        start = time.time()

        constraintStatUpper = Constraint(split_training_data, n_a, n_y, approximator=statistical_approximation, delta=delta,
                                         epsilon=epsilon, bound='upper')
        constraintTrue = Constraint(split_training_data, n_a, n_y, approximator=true_approximation, delta=delta, epsilon=epsilon)

        print("Initializing the constraint took {:.3f} seconds".format(time.time() - start))
        print("Initializing algorithms")
        algorithms = [
            ConstrainedGreedy(n_x, n_a, n_y, split_training_data, constraintStatUpper, statistical_approximation, name='Constrained Greedy', label='CG'),
            ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraintStatUpper, statistical_approximation, name='Constrained Dynamic Programming', label='CDPU'),
            ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraintTrue, true_approximation, name="Constrained Dynamic Programming True", label="CDPT"),
            NaiveGreedy(n_x, n_a, n_y, approximator=statistical_approximation, max_steps=n_a-1),
            NaiveDynamicProgramming(n_x, n_a, n_y, split_training_data, statistical_approximation, reward=reward)
        ]

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
            for j in range(n_test_samples):
                alg_evals.append(alg.evaluate(test_data[j]))
            evaluations[alg.name] = alg_evals
        print("Running Evaluate took {:.3f} seconds".format(time.time() - main_start))

        # Calculate % of population at max - treatment_slack treatment over time
        best_founds_efficiency = np.zeros((n_algorithms, n_a + 1))
        mean_num_tests = np.zeros(n_algorithms)
        for i_alg, alg in enumerate(algorithms):
            for i_sample in range(n_test_samples):
                treatments = evaluations[alg.name][i_sample]
                mean_num_tests[i_alg] += len(treatments)
                best_possible = np.max(test_data[i_sample][2])
                best_found = 0
                for i_treatment in range(len(best_founds_efficiency[i_alg])):
                    if i_treatment < len(treatments):
                        effect = treatments[i_treatment][1]
                        if effect > best_found:
                            best_found = effect
                    if best_found == best_possible:
                        best_founds_efficiency[i_alg][i_treatment] += 1
            best_founds_efficiency /= n_test_samples
            mean_num_tests /= n_test_samples
            efficiency_results[i_alg][i] = best_founds_efficiency[i_alg][n_a]
            mean_n_tests_results[i_alg][i] = mean_num_tests[i_alg]

    mean_effects = np.zeros(n_algorithms)
    mean_times = np.zeros(n_algorithms)
    var_effects = np.zeros(n_algorithms)
    var_times = np.zeros(n_algorithms)
    for i in range(n_algorithms):
        for j in range(n_tests):
            mean_effects[i] += efficiency_results[i][j]
            mean_times[i] += mean_n_tests_results[i][j]
    mean_effects /= n_tests
    mean_times /= n_tests
    for i in range(n_algorithms):
        for j in range(n_tests):
            var_effects[i] += (efficiency_results[i][j] - mean_effects[i])**2
            var_times[i] += (mean_n_tests_results[i][j] - mean_times[i])**2
    var_effects /= n_tests - 1
    var_times /= n_tests - 1

    for i in range(n_algorithms):
        print(algorithms[i].name)
        print("Mean effect: {}".format(mean_effects[i]))
        print("Mean times: {}".format(mean_times[i]))
        print("Var effect: {}".format(var_effects[i]))
        print("Var times: {}".format(var_times[i]))
print("Done!")
