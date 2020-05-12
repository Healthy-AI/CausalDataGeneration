from Algorithms.constrained_greedy import ConstrainedGreedy
from Algorithms.function_approximation import FunctionApproximation
from Algorithms.naive_dynamic_programming import NaiveDynamicProgramming
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.exact_approximator import ExactApproximator
from DataGenerator.data_generator import *
import time
from pathlib import Path
from Algorithms.better_treatment_constraint import Constraint
from Algorithms.statistical_approximator import StatisticalApproximator
from Database.antibioticsdatabase import AntibioticsDatabase


if __name__ == '__main__':
    # Training values
    #seed = 72991  # Used for both synthetic and real data
    seed = 284912491  # Used for both synthetic and real data
    n_z = 2
    n_x = 1
    n_a = 5
    n_y = 3
    n_test_samples = 2000
    data_limits = [10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 50000]
    n_training_samples = np.max(data_limits)
    epsilon = 0
    delta = 0.3

    # Plot values
    treatment_slack = 0     # Eg, how close to max must we be to be considered "good enough"
    plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
    plot_markers = ['s', 'v', 'P', '1', '2', '3', '4']
    plot_lines = ['-', '--', ':', '-.']
    plot_delta_grid_search = True
    delta_grid_search_percentage = True
    main_start = time.time()

    # Generate the data
    dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
    dist.print_moderator_statistics()
    dist.print_covariate_statistics()
    dist.print_treatment_statistics()
    dist.print_detailed_treatment_statistics()

    print("Generating test data set")
    test_data = generate_test_data(dist, n_test_samples)

    data_sets = []
    print("Generating training data set {}".format(n_training_samples))
    main_data = generate_data(dist, n_training_samples)
    for i in range(len(data_limits)):
        d_tmp = {}
        d_tmp['x'] = np.copy(main_data['x'][0:data_limits[i]])
        d_tmp['h'] = np.copy(main_data['h'][0:data_limits[i]])
        d_tmp['z'] = np.copy(main_data['z'][0:data_limits[i]])
        data_sets.append(split_patients(d_tmp))

    true_approximation = ExactApproximator(dist)
    evaluations_data_amount = {}
    time_name = 'time'
    outcome_name = 'outcome'
    for training_data_set in data_sets:
        print("Initializing approximator")
        statistical_approximationPrior = StatisticalApproximator(n_x, n_a, n_y, training_data_set, prior_mode='gaussian')
        statistical_approximationNone = StatisticalApproximator(n_x, n_a, n_y, training_data_set, prior_mode='none')
        function_approximation = FunctionApproximation(n_x, n_a, n_y, training_data_set)

        print("Initializing Constraint")
        start = time.time()
        constraintNone = Constraint(training_data_set, n_a, n_y, approximator=statistical_approximationNone, delta=delta, epsilon=epsilon)
        constraintPrior = Constraint(training_data_set, n_a, n_y, approximator=statistical_approximationPrior, delta=delta, epsilon=epsilon)
        constraintFunc = Constraint(training_data_set, n_a, n_y, approximator=function_approximation, delta=delta, epsilon=epsilon)
        constraintTrue = Constraint(training_data_set, n_a, n_y, approximator=true_approximation, delta=delta, epsilon=epsilon)

        print("Initializing the constraint took {:.3f} seconds".format(time.time()-start))
        print("Initializing algorithms")
        algorithms = [
            ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data_set, constraintNone, statistical_approximationNone, name="CDP_U", label="CDP_U"),
            ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data_set, constraintPrior, statistical_approximationPrior, name="CDP_H", label="CDP_H"),
            ConstrainedGreedy(n_x, n_a, n_y, training_data_set, constraintPrior, statistical_approximationPrior, name="CG_H"),
            ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data_set, constraintTrue, true_approximation, name="CDP_T", label="CDP_T"),
        ]

        n_algorithms = len(algorithms)

        for alg in algorithms:
            start = time.time()
            print("Training {}".format(alg.name))
            alg.learn()
            if alg.name not in evaluations_data_amount:
                evaluations_data_amount[alg.name] = {outcome_name: [], time_name: []}
            total_outcome = 0
            total_time = 0
            for i in range(n_test_samples):
                interventions = alg.evaluate(test_data[i])
                search_time = len(interventions)
                best_outcome = max([treatment[1] for treatment in interventions])
                if best_outcome == np.max(test_data[i][2]):
                    total_outcome += 1
                total_time += search_time
            mean_outcome = total_outcome / n_test_samples
            mean_time = total_time / n_test_samples
            evaluations_data_amount[alg.name][outcome_name].append(mean_outcome)
            evaluations_data_amount[alg.name][time_name].append(mean_time)
            print("Finished {} for data amount {} in {} seconds".format(alg.name, len(training_data_set['x']), time.time() - start))

    # Plot mean treatment effect vs delta
    fig, ax1 = plt.subplots(figsize=(10, 7))
    plt.title('Mean treatment effect/mean search time vs data amount (delta: {})'.format(delta))
    plt.xlabel('Data amount')
    ax2 = ax1.twinx()
    ax1.set_ylabel('Mean treatment effect')
    ax2.set_ylabel('Mean search time')
    lns = []
    for i_plot, alg_name in enumerate(evaluations_data_amount):
        ln1 = ax1.plot(data_limits, evaluations_data_amount[alg_name][outcome_name], plot_colors[i_plot],
                       label='{} {}'.format(alg_name, 'effect'))
        ln2 = ax2.plot(data_limits, evaluations_data_amount[alg_name][time_name], plot_colors[i_plot] + plot_lines[1],
                       label='{} {}'.format(alg_name, 'time'))
        lns.append(ln1)
        lns.append(ln2)
    plt.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.04, 0), loc='upper left')
    plt.xscale('log')
    plt.show(block=False)

    plt.show()