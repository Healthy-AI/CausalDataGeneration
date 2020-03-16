from Algorithms.q_learning import QLearner
from Algorithms.greedyShuffledHistory import GreedyShuffled
from Algorithms.greedyShuffledHistoryV2 import GreedyShuffled2
from Algorithms.constrained_q_learning import ConstrainedQlearner
from DataGenerator.data_generator import *
import time
import deepdish as dd
import random

# Training values
seed = 876
n_z = 6
n_x = 1
n_a = 7
n_y = 3
training_episodes = 50000
n_training_samples = 20000
n_test_samples = 20000
delta = 0
epsilon = 0
reward = 0.1

# Plot values
treatment_slack = 0     # Eg, how close to max must we be to be considered "good enough"
plot_colors = ['k', 'r', 'b', 'g']
plot_markers = ['', '--', ':']
main_start = time.time()

# Generate the data
dist = DiscreteDistribution(n_z, n_x, n_a, n_y, seed=seed)
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
    filename = '{}{}{}{}.h5'.format(dist.name, str(dataset['samples']), dataset['name'], seed)
    try:
        data = dd.io.load(filename)
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
        dd.io.save(filename, data)


split_training_data = datasets['training']['data']
test_data = datasets['test']['data']
print("Initializing algorithms")
algorithms = [
    GreedyShuffled(n_x, n_a, n_y, split_training_data, delta, epsilon),
    GreedyShuffled2(n_x, n_a, n_y, split_training_data, delta, epsilon),
    ConstrainedQlearner(n_x, n_a, n_y, split_training_data, delta=delta, epsilon=epsilon),
    QLearner(n_x, n_a, n_y, split_training_data, reward=-reward, learning_time=training_episodes,
             learning_rate=0.01, discount_factor=1)
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
for i, alg in enumerate(algorithms):
    for i_sample in range(n_test_samples):
        treatments = evaluations[alg.name][i_sample]
        best_found = 0
        for i_treatment in range(len(max_mean_treatment_effects[i])):
            if i_treatment < len(treatments):
                effect = treatments[i_treatment][1]
                if effect > best_found:
                    best_found = effect
            max_mean_treatment_effects[i][i_treatment] += best_found
max_mean_treatment_effects /= n_test_samples

# Calculate mean treatment effect over population
mean_treatment_effects = np.zeros((n_algorithms, n_a + 1))      # Overshoot by 1 to get all max values at last step
for i, alg in enumerate(algorithms):
    for i_sample in range(n_test_samples):
        treatments = evaluations[alg.name][i_sample]
        best_found = 0
        for i_treatment in range(len(mean_treatment_effects[i])):
            if i_treatment >= len(treatments):
                effect = best_found
            else:
                effect = treatments[i_treatment][1]
            if effect > best_found:
                best_found = effect
            mean_treatment_effects[i][i_treatment] += effect
mean_treatment_effects /= n_test_samples


# Plot mean treatment effect over population
x = np.arange(0, n_a+1)
axs1 = plt.subplot(121)
for i_plot, alg in enumerate(algorithms):
    plt.plot(x, mean_treatment_effects[i_plot], plot_colors[i_plot] + plot_markers[0], label=alg.label)
    plt.plot(x, max_mean_treatment_effects[i_plot], plot_colors[i_plot] + plot_markers[1])
    plt.fill_between(x, mean_treatment_effects[i_plot], max_mean_treatment_effects[i_plot], color=plot_colors[i_plot], alpha=0.1)
plt.grid(True)
average_max_treatment_effect = sum([max(data[-1]) for data in test_data])/len(test_data)
plt.plot(x, np.ones(len(x))*average_max_treatment_effect, label='MAX_AVG')
plt.legend(loc='lower right')


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
axs3 = plt.subplot(122)
for i_plot in range(n_algorithms):
    plt.plot(x, at_max[i_plot], plot_colors[i_plot])
plt.grid(True)
plt.show()
