from Algorithms.constrained_greedy import ConstrainedGreedy
from Algorithms.distribution_algorithm_wrapper import DistAlgWrapper
from Algorithms.function_approximation import FunctionApproximation
from Algorithms.naive_dynamic_programming import NaiveDynamicProgramming
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.naive_greedy import NaiveGreedy
from Algorithms.deep_q_learning import DeepQLearning
from Algorithms.true_approximator import TrueApproximator
from Algorithms.true_constraint import TrueConstraint
from DataGenerator.data_generator import *
import time
from pathlib import Path
from Algorithms.better_treatment_constraint import Constraint
from Algorithms.statistical_approximator import StatisticalApproximator
from Database.antibioticsdatabase import AntibioticsDatabase

if __name__ == '__main__':
    # Training values
    seed = None  # Used for both synthetic and real data
    n_z = 2
    n_x = 1
    n_a = 5
    n_y = 3
    training_episodes = 5000
    n_training_samples = 15000
    n_test_samples = 1000
    delta = 0.2
    epsilon = 0
    reward = -0.35
    # for grid search
    nr_deltas = 40
    delta_limit = 1

    # Plot values
    treatment_slack = 0  # Eg, how close to max must we be to be considered "good enough"
    plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
    plot_markers = ['s', 'v', 'P', '1', '2', '3', '4']
    plot_lines = [(i, (1, 4, 1, 4)) for i in range(0, 8)]
    alt_plot_lines = ['-', '--', ':', '-.']


    plot_mean_treatment_effect = False
    plot_treatment_efficiency = False
    plot_delta_efficiency = True
    plot_search_time = False
    plot_strictly_better = False
    plot_delta_grid_search = False
    delta_grid_search_percentage = False
    fixed_scale = False
    plotbools = [plot_mean_treatment_effect, plot_treatment_efficiency, plot_delta_efficiency, plot_search_time,
                 plot_strictly_better]
    main_start = time.time()

    # Generate the data

    #dist = DiscreteDistribution(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
    dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
    #dist = DiscreteDistributionWithInformation(n_z, n_x, n_a, n_y, seed=seed)
    #'''
    dist.print_moderator_statistics()
    dist.print_covariate_statistics()
    dist.print_treatment_statistics()
    dist.print_detailed_treatment_statistics()
    #'''
    #dist = AntibioticsDatabase(n_x=1, antibiotic_limit=5, seed=seed)
    '''
    dist = NewDistribution(seed=seed)
    #dist = NewDistributionSlightlyRandom(seed=seed)
    n_x = 1
    n_a = 3
    n_y = 3
    '''
    '''
    dist = FredrikDistribution()
    n_x = 1
    n_a = 3
    n_y = 2
    '''

    if type(dist) != AntibioticsDatabase:
        training = {'name': 'training', 'samples': n_training_samples, 'func': generate_data, 'split': True,
                    'database': False}
        test = {'name': 'test', 'samples': n_test_samples, 'func': generate_test_data, 'split': False,
                'database': False}
        datasets = {'training': training, 'test': test}

        for key, dataset in datasets.items():
            filename = '{}{}{}{}vars{}{}{}{}.h5'.format(
                dist.name, str(dataset['samples']), dataset['name'], seed,
                n_z, n_x, n_a, n_y)
            filepath = Path('Data', filename)
            '''
            try:
                data = dd.io.load(filepath)
                dataset['data'] = data
                print('Found %s data on file' % dataset['name'])
            except IOError:
            '''
            if True:
                start = time.time()
                n_samples = dataset['samples']
                print("Generating {} {} samples...".format(n_samples, dataset['name']))
                generate_data_func = dataset['func']
                if dataset['database']:
                    data = generate_data_func(n_samples)
                else:
                    data = generate_data_func(dist, n_samples)
                if dataset['split']:
                    data = split_patients(data)
                print("Generating samples took {:.3f} seconds".format(time.time() - start))
                dataset['data'] = data
                # if seed is not None:
                #    dd.io.save(filepath, data)
    else:
        training_data, test_data = dist.get_data()
        datasets = {'training': {'data': split_patients(training_data)}, 'test': {'data': test_data}}
        n_x = dist.n_x
        n_a = dist.n_a
        n_y = dist.n_y
        n_test_samples = len(test_data)

    split_training_data = datasets['training']['data']
    test_data = datasets['test']['data']
    # print("Initializing function approximator")
    # start = time.time()
    function_approximation = FunctionApproximation(n_x, n_a, n_y, split_training_data)
    # print("Initializing {} took {:.3f} seconds".format(function_approximation.name, time.time()-start))
    print("Initializing statistical approximator")
    start = time.time()

    statistical_approximation = StatisticalApproximator(n_x, n_a, n_y, split_training_data, prior_mode='gaussian')
    #print("Initializing {} took {:.3f} seconds".format(statistical_approximation.name, time.time() - start))

    true_approximation = TrueApproximator(dist)

    print("Initializing Constraint")
    start = time.time()

    constraintStat = Constraint(split_training_data, n_a, n_y, approximator=statistical_approximation, delta=delta, epsilon=epsilon)
    constraintTrue = Constraint(split_training_data, n_a, n_y, approximator=true_approximation, delta=delta, epsilon=epsilon)
    constraintCT = TrueConstraint(dist, approximator=statistical_approximation, delta=delta, epsilon=epsilon)
    constraintTT = TrueConstraint(dist, approximator=true_approximation, delta=delta, epsilon=epsilon)
    constraintFuncApprox = Constraint(split_training_data, n_a, n_y, approximator=function_approximation, delta=delta, epsilon=epsilon)

    print("Initializing the constraint took {:.3f} seconds".format(time.time() - start))
    print("Initializing algorithms")
    algorithms = [
        #GreedyShuffled(n_x, n_a, n_y, split_training_data, delta, epsilon),
        #ConstrainedGreedy(n_x, n_a, n_y, split_training_data, constraintTrue, true_approximation, name="Constrained Greedy True", label="CGT"),
        #ConstrainedGreedy(n_x, n_a, n_y, split_training_data, constraintStat, statistical_approximation),
        #ConstrainedGreedy(n_x, n_a, n_y, split_training_data, constraintFuncApprox, function_approximation, name="Constrained Greedy FuncApprox"),
        ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraintTrue, true_approximation, name="Dynamic Programming True", label="CDPT"),
        ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraintStat, statistical_approximation),
        #ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraintStat, function_approximation, name="Dynamic Programming Func", label="CDPF"),
        #ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraintFuncApprox, function_approximation,name="Constrained Dynamic Programming FuncApprox"),
        #NaiveGreedy(n_x, n_a, n_y, split_training_data),
        #DistAlgWrapper(dist, name="Distribution", label="dist"),
        #NaiveDynamicProgramming(n_x, n_a, n_y, split_training_data, statistical_approximation, reward=reward)
        #QLearner(n_x, n_a, n_y, split_training_data, reward=reward, learning_time=training_episodes, learning_rate=0.01, discount_factor=1),
        #QLearnerConstrained(n_x, n_a, n_y, split_training_data, constraint, learning_time=training_episodes, learning_rate=0.01, discount_factor=1),
        #OnlineQLearner(n_x, n_a, n_y, dist, constraint, learning_time=training_episodes),
        #DeepQLearning(n_x, n_a, n_y, split_training_data, constraint=constraintFuncApprox,  approximator=function_approximation),
        #DeepQLearning(n_x, n_a, n_y, split_training_data, constraint=constraintStat, approximator=statistical_approximation)
    ]

    n_algorithms = len(algorithms)
    if any(plotbools):
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
        print("Running Evaluate took {:.3f} seconds".format(time.time() - main_start))

    print("Showing plots...")
    if plot_mean_treatment_effect or plot_search_time:
        # Calculate max mean treatment effect over population
        max_mean_treatment_effects = np.zeros((n_algorithms, n_a))
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
        max_mean_treatment_effects /= n_test_samples

        # Calculate mean treatment effect over population
        mean_treatment_effects = np.zeros((n_algorithms, n_a + 1))  # Overshoot by 1 to get all max values at last step
        mean_num_tests = np.zeros(n_algorithms)

        for i_sample in range(n_test_samples):
            for i_alg, alg in enumerate(algorithms):
                treatments = evaluations[alg.name][i_sample]
                mean_num_tests[i_alg] += len(treatments)
                best_found = 0
                for i_treatment in range(len(mean_treatment_effects[i_alg])):
                    if i_treatment >= len(treatments):
                        effect = best_found
                    else:
                        effect = treatments[i_treatment][1]
                    if effect > best_found:
                        best_found = effect
                    mean_treatment_effects[i_alg][i_treatment] += effect
        mean_treatment_effects /= n_test_samples
        mean_num_tests /= n_test_samples

        if plot_mean_treatment_effect:
            # Plot mean treatment effect over population
            x = np.arange(0, n_a)
            x_ticks = list(np.arange(1, n_a + 1))
            plt.figure()
            plt.title('Treatment effect. delta: {}'.format(delta))
            plt.ylabel('Mean treatment effect')
            plt.xlabel('Number of tried treatments')
            average_max_treatment_effect = sum([max(data[-1]) for data in test_data]) / len(test_data)
            mean_lines = np.linspace(0, 1, n_algorithms + 1)
            for i_plot, alg in enumerate(algorithms):
                plt.plot(x, max_mean_treatment_effects[i_plot],
                         plot_markers[i_plot] + plot_colors[i_plot], linestyle=plot_lines[i_plot % len(plot_lines)],
                         linewidth=4, label=alg.label)
                plt.plot(x, max_mean_treatment_effects[i_plot], plot_colors[i_plot], linestyle='-', linewidth=2,
                         alpha=0.3)
                # plt.plot(x, mean_treatment_effects[i_plot], plot_markers[i_plot] + plot_colors[i_plot] + plot_lines[1])
                # plt.fill_between(x, mean_treatment_effects[i_plot], max_mean_treatment_effects[i_plot], color=plot_colors[i_plot], alpha=0.1)
                plt.axvline(mean_num_tests[i_plot] - 1, ymin=mean_lines[i_plot], ymax=mean_lines[i_plot + 1],
                            color=plot_colors[i_plot])
                plt.axvline(mean_num_tests[i_plot] - 1, ymin=0, ymax=1,
                            color=plot_colors[i_plot], alpha=0.1)

            plt.grid(True)
            plt.xticks(x, x_ticks)
            plt.plot(x, np.ones(len(x)) * average_max_treatment_effect, linestyle=plot_lines[-1], label='MAX_POSS_AVG')

            plt.legend(loc='lower right')
            plt.show(block=False)

    if plot_delta_efficiency:
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

        # Plot mean treatment effect over population
        plt.figure()
        plt.title('Treatment efficiency. d: {}'.format(delta))
        plt.ylabel('Percentage of population at best possible treatment')
        plt.xlabel('Number of tried treatments')
        x = np.arange(0, n_a + 1)
        x_ticks = list(np.arange(1, n_a + 2))
        x_ticks[-1] = 'Done'
        for i_plot, alg in enumerate(algorithms):
            plt.plot(x, best_founds_efficiency[i_plot], plot_markers[i_plot] + plot_colors[i_plot] + alt_plot_lines[0],
                     label=alg.label)
            plt.axvline(mean_num_tests[i_plot] - 1, 0, 1, color=plot_colors[i_plot])
        plt.xticks(x, x_ticks)
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.show(block=False)

    if plot_search_time:
        # Plot mean number of treatments tried
        plt.figure()
        plt.title('Search time')
        plt.ylabel('Mean number of treatments tried')
        plt.xlabel('Policy')
        x_bars = []
        for i_alg, alg in enumerate(algorithms):
            x_bars.append(alg.name)
        x_bars = [label.replace(" ", '\n') for label in x_bars]
        rects = plt.bar(x_bars, mean_num_tests)
        for rect in rects:
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 0.90 * h, "%f" % h, ha="center", va="bottom")
        plt.show(block=False)

    if plot_strictly_better:
        # Find strictly better samples for each algorithm
        strictly_better_samples = np.zeros(n_algorithms, dtype=int)
        for i_sample in range(n_test_samples):
            samples = np.zeros((n_algorithms, 2))
            for i_alg, alg in enumerate(algorithms):
                treatments = evaluations[alg.name][i_sample]
                n_treatments = len(treatments)
                best_found_outcome = max([intervention[1] for intervention in treatments])
                samples[i_alg, 0] = n_treatments
                samples[i_alg, 1] = best_found_outcome

            min_treatments = np.where(samples[:, 0] == samples[:, 0].min())
            max_outcome = np.where(samples[:, 1] == samples[:, 1].max())
            strictly_better_indices, _, _ = np.intersect1d(min_treatments, max_outcome, return_indices=True)
            if len(strictly_better_indices) == 1:
                strictly_better_samples[strictly_better_indices[0]] += 1

        for i_alg, alg in enumerate(algorithms):
            print(alg.name, 'has', strictly_better_samples[i_alg],
                  'strictly better samples than the other algorithms')
        print('There is a total of', n_test_samples, 'test samples')

        # Plot strictly better samples statistics
        plt.figure()
        plt.title('% of strictly better samples')
        plt.ylabel('% of samples where the policy performed better')
        plt.xlabel('Policy')
        x_bars = []
        for i_alg, alg in enumerate(algorithms):
            x_bars.append(alg.name)
        x_bars = [label.replace(" ", '\n') for label in x_bars]
        rects = plt.bar(x_bars, (strictly_better_samples / n_test_samples) * 100)
        for rect in rects:
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 0.90 * h, "%f" % h, ha="center", va="bottom")
        plt.show(block=False)

    if plot_delta_grid_search:
        time_name = 'time'
        outcome_name = 'outcome'
        evaluations_delta = {}
        deltas = np.linspace(0, delta_limit, nr_deltas)
        for delta in deltas:
            for alg in algorithms:
                if alg.name not in evaluations_delta:
                    evaluations_delta[alg.name] = {outcome_name: [], time_name: []}
                try:
                    alg.constraint.delta = delta
                    alg.constraint.better_treatment_constraint_dict = {}
                except AttributeError:
                    pass
                print("Training {}".format(alg.name))
                alg.learn()
                total_outcome = 0
                total_time = 0
                print("Evaluating {} with delta = {}".format(alg.name, delta))
                for i in range(n_test_samples):
                    interventions = alg.evaluate(test_data[i])
                    search_time = len(interventions)
                    best_outcome = max([treatment[1] for treatment in interventions])
                    if delta_grid_search_percentage:
                        if best_outcome == np.max(test_data[i][2]):
                            total_outcome += 1
                    else:
                        total_outcome += best_outcome
                    total_time += search_time
                mean_outcome = total_outcome / n_test_samples
                mean_time = total_time / n_test_samples
                evaluations_delta[alg.name][outcome_name].append(mean_outcome)
                evaluations_delta[alg.name][time_name].append(mean_time)
        print("Running Evaluate (and training) over delta took {:.3f} seconds".format(time.time() - main_start))

        # Plot mean treatment effect vs delta
        fig, ax1 = plt.subplots(figsize=(10, 7))
        plt.title('Mean treatment effect/mean search time vs delta')
        plt.xlabel('delta')
        ax2 = ax1.twinx()
        ax1.set_ylabel('Mean treatment effect')
        ax2.set_ylabel('Mean search time')
        lns = []
        for i_plot, alg in enumerate(algorithms):
            ln1 = ax1.plot(deltas, evaluations_delta[alg.name][outcome_name], plot_colors[i_plot] + alt_plot_lines[0],
                           label='{} {}'.format(alg.label, 'effect'))
            ln2 = ax2.plot(deltas, evaluations_delta[alg.name][time_name], plot_colors[i_plot] + alt_plot_lines[1],
                           label='{} {}'.format(alg.label, 'time'))
            lns.append(ln1)
            lns.append(ln2)
        average_max_treatment_effect = sum([max(data[-1]) for data in test_data]) / len(test_data)
        if delta_grid_search_percentage:
            ax1.plot(deltas, np.ones(nr_deltas) * 1, alt_plot_lines[3], label='MAX_POSS_AVG')
        else:
            ax1.plot(deltas, np.ones(nr_deltas) * average_max_treatment_effect, alt_plot_lines[3], label='MAX_POSS_AVG')
        plt.grid(True)
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 1, box.height])

        # Put a legend to the right of the current axis

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.04, 0), loc='upper left')
        # plt.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
        plt.show(block=False)

    plt.show()
print("Done!")
