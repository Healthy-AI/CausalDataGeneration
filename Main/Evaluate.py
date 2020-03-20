from Algorithms.q_learning import QLearner
from Algorithms.q_learning_with_constraint import QLearnerConstrained
from Algorithms.greedyShuffledHistory import GreedyShuffled
from Algorithms.greedyShuffledHistoryV2 import GreedyShuffled2
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from DataGenerator.data_generator import *
import time
import deepdish as dd
import random
from pathlib import Path
from Algorithms.online_q_learning import OnlineQLearner

# Training values
seed = 8956
n_z = 4
n_x = 3
n_a = 5
n_y = 5
training_episodes = 100000
n_training_samples = 20000
n_test_samples = 2000
delta = 0.2
epsilon = 0
reward = -0.25

# Plot values
treatment_slack = 0     # Eg, how close to max must we be to be considered "good enough"
plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
plot_markers = ['', '--', ':']
main_start = time.time()

# Generate the data
dist = DiscreteDistribution(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)

'''
dist = NewDistribution(seed=seed)
n_x = 1
n_a = 3
n_y = 3
'''

training = {'name': 'training', 'samples': n_training_samples, 'func': generate_data, 'split': True}
test = {'name': 'test', 'samples': n_test_samples, 'func': generate_test_data, 'split': False}
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
        data = generate_data_func(dist, n_samples)
        if dataset['split']:
            data = split_patients(data)
        print("Generating samples took {:.3f} seconds".format(time.time()-start))
        dataset['data'] = data
        if seed is not None:
            dd.io.save(filepath, data)


split_training_data = datasets['training']['data']
test_data = datasets['test']['data']
print("Initializing algorithms")
algorithms = [
    #GreedyShuffled(n_x, n_a, n_y, split_training_data, delta, epsilon),
    GreedyShuffled2(n_x, n_a, n_y, split_training_data, delta=delta, epsilon=epsilon),
    ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, delta=delta, epsilon=epsilon),
    #QLearner(n_x, n_a, n_y, split_training_data, reward=reward, learning_time=training_episodes, learning_rate=0.01, discount_factor=1),
    QLearnerConstrained(n_x, n_a, n_y, split_training_data, delta=delta, epsilon=epsilon, learning_time=training_episodes, learning_rate=0.01, discount_factor=1),
    #OnlineQLearner(n_x, n_a, n_y, dist, learning_time=training_episodes),
]

n_algorithms = len(algorithms)

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
    plt.plot(x, mean_treatment_effects[i_plot], plot_colors[i_plot] + plot_markers[0], label=alg.label)
    plt.plot(x, max_mean_treatment_effects[i_plot], plot_colors[i_plot] + plot_markers[1])
    plt.fill_between(x, mean_treatment_effects[i_plot], max_mean_treatment_effects[i_plot], color=plot_colors[i_plot], alpha=0.1)
    plt.axvline(mean_num_tests[i_plot], 0, average_max_treatment_effect, color=plot_colors[i_plot])

plt.grid(True)
plt.xticks(x, x_ticks)
plt.plot(x, np.ones(len(x))*average_max_treatment_effect, label='MAX_POSS_AVG')

plt.legend(loc='lower right')
plt.show(block=False)


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
for i_plot, alg in enumerate(algorithms):
    plt.plot(x, at_max[i_plot], plot_colors[i_plot], label=alg.label)
plt.xticks(x, x_ticks)
plt.grid(True)
plt.legend(loc='lower right')
plt.show(block=False)

# Plot mean number of treatments tried
plt.figure()
plt.title('Search time')
plt.ylabel('Mean number of treatments tried')
plt.xlabel('Policy')
x_bars = []
for i_alg, alg in enumerate(algorithms):
    x_bars.append(alg.name)
rects = plt.bar(x_bars, mean_num_tests)
for rect in rects:
    h = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., 0.90*h, "%f" % h, ha="center", va="bottom")
plt.show(block=False)

plt.show()