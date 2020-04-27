from Algorithms.function_approximation import FunctionApproximation
from Algorithms.naive_dynamic_programming import NaiveDynamicProgramming
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.naive_greedy import NaiveGreedy
from Algorithms.deep_q_learning import DeepQLearning
from Algorithms.constrained_greedy import ConstrainedGreedy
from Algorithms.true_approximator import TrueApproximator
from DataGenerator.data_generator import *
import time
from pathlib import Path
from Algorithms.better_treatment_constraint import Constraint
from Algorithms.statistical_approximator import StatisticalApproximator
from Database.antibioticsdatabase import AntibioticsDatabase

if __name__ == '__main__':
    # Training values
    #seed = 90821  # Used for both synthetic and real data
    n_tests = 10
    seeds = [np.random.randint(0, 10000) for i in range(n_tests)]
    print(seeds)
    n_z = 3
    n_x = 1
    n_a = 5
    n_y = 3
    training_episodes = 750000
    n_training_samples = 30000
    n_test_samples = 3000
    delta = 0.0
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
    plot_delta_efficiency = False
    plot_search_time = False
    plot_strictly_better = False
    plot_delta_grid_search = False
    delta_grid_search_percentage = False
    fixed_scale = False
    plotbools = [plot_mean_treatment_effect, plot_treatment_efficiency, plot_delta_efficiency, plot_search_time,
                 plot_strictly_better]
    main_start = time.time()

    n_algorithms = 4
    mean_n_tests_results = np.zeros((n_tests, n_algorithms))
    efficiency_results = np.zeros((n_tests, n_algorithms))
    for i, seed in enumerate(seeds):
        # Generate the data
        # dist = DiscreteDistribution(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
        dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
        # dist = DiscreteDistributionWithInformation(n_z, n_x, n_a, n_y, seed=seed)
        #'''
        dist.print_moderator_statistics()
        dist.print_covariate_statistics()
        dist.print_treatment_statistics()
        dist.print_detailed_treatment_statistics()
        #'''
        #dist = AntibioticsDatabase(n_x=2, antibiotic_limit=6, seed=seed)

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
        statistical_approximation = StatisticalApproximator(n_x, n_a, n_y, split_training_data, prior_mode='none')
        # print("Initializing {} took {:.3f} seconds".format(statistical_approximation.name, time.time() - start))

        true_approximation = TrueApproximator(dist)

        print("Initializing Constraint")
        start = time.time()

        constraintStatUpper = Constraint(split_training_data, n_a, n_y, approximator=statistical_approximation, delta=delta,
                                         epsilon=epsilon, bound='upper')
       # constraintStatLower = Constraint(split_training_data, n_a, n_y, approximator=statistical_approximation, delta=delta,
       #                                  epsilon=epsilon, bound='lower')
        # constraintTrue = Constraint(split_training_data, n_a, n_y, approximator=true_approximation, delta=delta, epsilon=epsilon)
        constraintFuncApprox = Constraint(split_training_data, n_a, n_y, approximator=function_approximation, delta=delta,
                                          epsilon=epsilon)

        print("Initializing the constraint took {:.3f} seconds".format(time.time() - start))
        print("Initializing algorithms")
        algorithms = [
            ConstrainedGreedy(n_x, n_a, n_y, split_training_data, constraintStatUpper, statistical_approximation,
                               name='Constrained Greedy', label='CG'),
            ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraintStatUpper, statistical_approximation,
                                          name='Constrained Dynamic Programming Upper', label='CDPU'),
            NaiveGreedy(n_x, n_a, n_y, split_training_data),
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
        efficiency_results[i] = best_founds_efficiency
        mean_n_tests_results[i] = mean_num_tests
    print([alg.name for alg in algorithms])
    print(mean_n_tests_results)
    print(efficiency_results)

print("Done!")
