from Algorithms.naive_greedy import NaiveGreedy
from Algorithms.q_learning import QLearner
from Algorithms.q_learning_with_constraint import QLearnerConstrained
from Algorithms.greedyShuffledHistory import GreedyShuffled
from Algorithms.constrained_greedy import ConstrainedGreedy
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.true_approximator import TrueApproximator
from DataGenerator.data_generator import *
import time
import deepdish as dd
import random
from pathlib import Path
from Algorithms.online_q_learning import OnlineQLearner
from Algorithms.better_treatment_constraint import Constraint
from Algorithms.function_approximation import FunctionApproximation
from Algorithms.statistical_approximator import StatisticalApproximator
from Database.antibioticsdatabase import AntibioticsDatabase

if __name__ == '__main__':
    # Training values
    seed = 1337  # Used for both synthetic and real data
    n_z = 2
    n_x = 1
    n_a = 5
    n_y = 3
    training_episodes = 750000
    n_training_samples = 20000
    n_test_samples = 2000
    delta = 0.15
    epsilon = 0
    reward = -0.25
    # for grid search
    nr_deltas = 5
    delta_limit = 1

    # Plot values
    treatment_slack = 0     # Eg, how close to max must we be to be considered "good enough"
    plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
    plot_markers = ['s', 'v', 'P', '1', '2', '3', '4']
    plot_lines = ['-', '--', ':', '-.']
    plot_mean_treatment_effect = True
    plot_treatment_efficiency = False
    plot_search_time = False
    plot_strictly_better = False
    plot_delta_grid_search = False
    plotbools = [plot_mean_treatment_effect, plot_treatment_efficiency, plot_search_time, plot_strictly_better]
    main_start = time.time()

    # Generate the data
    #dist = DiscreteDistribution(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
    #dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
    dist = DiscreteDistributionWithInformation(n_z, n_x, n_a, n_y, seed=seed)
    dist.print_moderator_statistics()
    dist.print_covariate_statistics()
    dist.print_treatment_statistics()
    dist.print_detailed_treatment_statistics()
    #dist = AntibioticsDatabase(seed=seed)
    '''
    dist = NewDistributionSlightlyRandom(seed=seed)
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
        training = {'name': 'training', 'samples': n_training_samples, 'func': generate_data, 'split': True, 'database': False}
        test = {'name': 'test', 'samples': n_test_samples, 'func': generate_test_data, 'split': False, 'database': False}
        datasets = {'training': training, 'test': test}

        for key, dataset in datasets.items():
            filename = '{}{}{}{}vars{}{}{}{}.h5'.format(
                dist.name, str(dataset['samples']), dataset['name'], seed,
                n_z, n_x, n_a, n_y)
            filepath = Path('Data', filename)

            try:
                data = dd.io.load(filepath)
                dataset['data'] = data
                print('Found %s data on file' % dataset['name'])
            except IOError:
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
                print("Generating samples took {:.3f} seconds".format(time.time()-start))
                dataset['data'] = data
                if seed is not None:
                    dd.io.save(filepath, data)
    else:
        datasets = {'training': {'data': dist.get_data()}, 'test': {'data': dist.get_test_data(n_test_samples)}}

        n_x = dist.n_x
        n_a = dist.n_a
        n_y = dist.n_y

    split_training_data = split_patients(datasets['training']['data'])
    test_data = datasets['test']['data']
    print("Initializing function approximator")
    start = time.time()
    function_approximation = FunctionApproximation(n_x, n_a, n_y, split_training_data)
    print("Initializing {} took {:.3f} seconds".format(function_approximation.name, time.time()-start))
    print("Initializing statistical approximator")
    start = time.time()
    statistical_approximation = StatisticalApproximator(n_x, n_a, n_y, split_training_data)
    print("Initializing {} took {:.3f} seconds".format(statistical_approximation.name, time.time() - start))

    true_approximation = TrueApproximator(dist)

    print("Initializing Constraint")
    start = time.time()
    constraintStat = Constraint(split_training_data, n_a, n_y, approximator=statistical_approximation, delta=delta, epsilon=epsilon)
    constraintTrue = Constraint(split_training_data, n_a, n_y, approximator=true_approximation, delta=delta, epsilon=epsilon)
    print("Initializing the constraint took {:.3f} seconds".format(time.time()-start))
    print("Initializing algorithms")
    algorithms = [
        #GreedyShuffled(n_x, n_a, n_y, split_training_data, delta, epsilon),
        #ConstrainedGreedy(n_x, n_a, n_y, split_training_data, constraintTrue, true_approximation, name="Constrained Greedy True", label="CGT"),
        ConstrainedGreedy(n_x, n_a, n_y, split_training_data, constraintStat, statistical_approximation),
        #ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraintTrue, true_approximation, name="Dynamic Programming True", label="CDPT"),
        ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraintStat, statistical_approximation),
        NaiveGreedy(n_x, n_a, n_y, split_training_data),
        #QLearner(n_x, n_a, n_y, split_training_data, reward=reward, learning_time=training_episodes, learning_rate=0.01, discount_factor=1),
        #QLearnerConstrained(n_x, n_a, n_y, split_training_data, constraint, learning_time=training_episodes, learning_rate=0.01, discount_factor=1),
        #OnlineQLearner(n_x, n_a, n_y, dist, constraint, learning_time=training_episodes),
    ]

    n_algorithms = len(algorithms)
    if any(plotbools):
        for alg in algorithms:
            # Train the algorithms
            start = time.time()
            print("\tTraining %s..." % alg.name)
            alg.learn()
            print("\tTraining the %s algorithm took {:.3f} seconds".format(time.time()-start) % alg.name)

        # Evaluate the algorithms
        evaluations = {}
        for alg in algorithms:
            alg_evals = []
            print("Evaluating {}".format(alg.name))
            for i in range(n_test_samples):
                alg_evals.append(alg.evaluate(test_data[i]))
            evaluations[alg.name] = alg_evals
        print("Running Evaluate took {:.3f} seconds".format(time.time()-main_start))

    print("Showing plots...")
    if plot_mean_treatment_effect or plot_search_time:
        # Calculate max mean treatment effect over population
        max_mean_treatment_effects = np.zeros((n_algorithms, n_a + 1))
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
        mean_treatment_effects = np.zeros((n_algorithms, n_a + 1))      # Overshoot by 1 to get all max values at last step
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
            x = np.arange(0, n_a+1)
            x_ticks = list(np.arange(1, n_a+2))
            x_ticks[-1] = 'Done'
            plt.figure()
            plt.title('Treatment effect')
            plt.ylabel('Mean treatment effect')
            plt.xlabel('Number of tried treatments')
            average_max_treatment_effect = sum([max(data[-1]) for data in test_data])/len(test_data)
            for i_plot, alg in enumerate(algorithms):
                plt.plot(x, max_mean_treatment_effects[i_plot], plot_markers[i_plot] + plot_colors[i_plot] + plot_lines[0])
                plt.plot(x, mean_treatment_effects[i_plot], plot_markers[i_plot] + plot_colors[i_plot] + plot_lines[1], label=alg.label)
                plt.fill_between(x, mean_treatment_effects[i_plot], max_mean_treatment_effects[i_plot], color=plot_colors[i_plot], alpha=0.1)
                plt.axvline(mean_num_tests[i_plot]-1, 0, average_max_treatment_effect, color=plot_colors[i_plot])

            plt.grid(True)
            plt.xticks(x, x_ticks)
            plt.plot(x, np.ones(len(x)) * average_max_treatment_effect, plot_lines[3], label='MAX_POSS_AVG')

            plt.legend(loc='lower right')
            plt.show(block=False)

    if plot_treatment_efficiency:
        # Calculate % of population at max - treatment_slack treatment over time
        max_treatments = np.zeros(n_test_samples)
        for i_sample in range(n_test_samples):
            max_treatments[i_sample] = max(test_data[i_sample][2])
        at_max = np.zeros((n_algorithms, n_a + 1))
        for i, alg in enumerate(algorithms):
            for i_sample in range(n_test_samples):
                treatments = evaluations[alg.name][i_sample]
                found_max = 0
                for i_treatment in range(len(at_max[i])):
                    if i_treatment >= len(treatments):
                        at_max[i][i_treatment] += found_max
                    else:
                        if max_treatments[i_sample] <= treatments[i_treatment][1] + treatment_slack:
                            at_max[i][i_treatment] += 1
                            found_max = 1
        at_max /= n_test_samples


        # Plot mean treatment effect over population
        plt.figure()
        plt.title('Treatment efficiency')
        plt.ylabel('Percentage of population at best possible treatment')
        plt.xlabel('Number of tried treatments')
        x = np.arange(0, n_a + 1)
        x_ticks = list(np.arange(1, n_a + 2))
        x_ticks[-1] = 'Done'
        for i_plot, alg in enumerate(algorithms):
            plt.plot(x, at_max[i_plot], plot_colors[i_plot], label=alg.label)
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
            plt.text(rect.get_x() + rect.get_width()/2., 0.90*h, "%f" % h, ha="center", va="bottom")
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
        rects = plt.bar(x_bars, (strictly_better_samples/n_test_samples)*100)
        for rect in rects:
            h = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., 0.90*h, "%f" % h, ha="center", va="bottom")
        plt.show(block=False)

    if plot_delta_grid_search:
        time_name = 'time'
        outcome_name = 'outcome'
        evaluations_delta = {}
        deltas = np.linspace(0, delta_limit, nr_deltas)
        for delta in deltas:
            constraint.better_treatment_constraint_dict = {}
            constraint.delta = delta
            for alg in algorithms:
                if alg.name not in evaluations_delta:
                    evaluations_delta[alg.name] = {outcome_name: [], time_name: []}
                try:
                    alg.constraint = constraint
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
                    total_outcome += best_outcome
                    total_time += search_time
                mean_outcome = total_outcome/n_test_samples
                mean_time = total_time/n_test_samples
                evaluations_delta[alg.name][outcome_name].append(mean_outcome)
                evaluations_delta[alg.name][time_name].append(mean_time)
        print("Running Evaluate (and training) over delta took {:.3f} seconds".format(time.time() - main_start))

        # Plot mean treatment effect vs delta
        fig, ax1 = plt.subplots()
        plt.title('Mean treatment effect/mean search time vs delta')
        plt.xlabel('delta')
        ax2 = ax1.twinx()
        ax1.set_ylabel('Mean treatment effect')
        ax2.set_ylabel('Mean search time')
        lns = []
        for i_plot, alg in enumerate(algorithms):
            ln1 = ax1.plot(deltas, evaluations_delta[alg.name][outcome_name], plot_colors[i_plot],
                           label='{} {}'.format(alg.label, 'effect'))
            ln2 = ax2.plot(deltas, evaluations_delta[alg.name][time_name], plot_colors[i_plot] + plot_lines[1],
                           label='{} {}'.format(alg.label, 'time'))
            lns.append(ln1)
            lns.append(ln2)
        average_max_treatment_effect = sum([max(data[-1]) for data in test_data]) / len(test_data)
        ax1.plot(deltas, np.ones(nr_deltas) * average_max_treatment_effect, plot_lines[3], label='MAX_POSS_AVG')
        plt.grid(True)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2)
        plt.show(block=False)

    plt.show()

