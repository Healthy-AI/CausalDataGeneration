from Algorithms.q_learning import QLearner
from Algorithms.greedyShuffledHistory import GreedyShuffled
from DataGenerator.distributions import *
import Algorithms.greedyShuffledHistory as greedy
import Algorithms.constrained_q_learning as cql
import matplotlib.pyplot as plt
from DataGenerator.data_generator import *
import Algorithms.q_learning as ql
import time

# Training values
seed = 12345
n_z = 6
n_x = 1
n_a = 7
n_y = 3
n_algorithms = 3
training_episodes = 50000
n_training_samples = 2000
n_test_samples = 2000
delta = 0.1
epsilon = 0

# Plot values
treatment_slack = 0     # Eg, how close to max must we be to be considered "good enough"
plot_colors = ['k', 'r', 'b', 'g']
plot_markers = ['', '--', ':']
plot_labels = ['QL', 'G', 'CQL']
main_start = time.time()

# Generate the data
dist = DiscreteDistribution(n_z, n_x, n_a, n_y, seed=seed)
#dist = NewDistribution(seed=seed)
start = time.time()
print("Generating {} training samples...".format(n_training_samples))
training_data = generate_data(dist, n_training_samples)
split_training_data = split_patients(training_data.copy())
print("Generating training samples took {:.3f} seconds".format(time.time()-start))
start = time.time()
print("Generating {} test samples...".format(n_test_samples))
test_data = generate_test_data(dist, n_test_samples)
print("Generating test samples took {:.3f} seconds".format(time.time()-start))

# Initialize and train the algorithms
start = time.time()
print("Initializing Greedy...")
G = greedy.GreedyShuffled(n_x, n_y, n_a)
print("\tTraining Greedy...")
G.find_probabilities(training_data)
print("\tTraining the Greedy algorithm took {:.3f} seconds".format(time.time()-start))

start = time.time()
print("Initializing Constrained Q-learning...")
CQL = cql.ConstrainedQlearner(n_x, n_a, n_y, split_training_data)
print("\tTraining Constrained Q-learning...")
CQL.learn()
print("\tTraining the Constrained Q-learning algorithm took {:.3f} seconds".format(time.time()-start))

start = time.time()
print("Initializing Q-learning...")
QL = ql.QLearner(n_x, n_a, n_y, split_training_data, reward=-delta, learning_time=training_episodes, learning_rate=0.01, discount_factor=1)
print("\tTraining Q-learning...")
QL.learn()
print("\tTraining the Q-learning algorithm took {:.3f} seconds".format(time.time()-start))

# Evaluate the algorithms
evaluations = [[], [], []]
for i in range(n_test_samples):
    # evaluations[1].append(CQL.evaluate(test_data[i]))
    evaluations[0].append(QL.evaluate(test_data[i]))
    evaluations[1].append(G.evaluate(test_data[i], delta, epsilon))
    evaluations[2].append(CQL.evaluate(test_data[i]))
print("Running Evaluate took {:.3f} seconds".format(time.time()-main_start))

print("Showing plots...")
# Calculate max mean treatment effect over population
max_mean_treatment_effects = np.zeros((n_algorithms, n_a + 1))
for i_alg in range(n_algorithms):
    for i_sample in range(n_test_samples):
        treatments = evaluations[i_alg][i_sample]
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
for i_alg in range(n_algorithms):
    for i_sample in range(n_test_samples):
        treatments = evaluations[i_alg][i_sample]
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


# Plot mean treatment effect over population
x = np.arange(0, n_a+1)
axs1 = plt.subplot(121)
for i_plot in range(n_algorithms):
    plt.plot(x, mean_treatment_effects[i_plot], plot_colors[i_plot] + plot_markers[0], label=plot_labels[i_plot])
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
for i_alg in range(n_algorithms):
    for i_sample in range(n_test_samples):
        treatments = evaluations[i_alg][i_sample]
        found_max = 0
        for i_treatment in range(len(at_max[i_alg])):
            if i_treatment >= len(treatments):
                at_max[i_alg][i_treatment] += found_max
            else:
                if max_treatments[i_sample] <= treatments[i_treatment][1] + treatment_slack:
                    at_max[i_alg][i_treatment] += 1
                    found_max = 1
at_max /= n_test_samples
# Plot mean treatment effect over population
axs3 = plt.subplot(122)
for i_plot in range(n_algorithms):
    plt.plot(x, at_max[i_plot], plot_colors[i_plot])
plt.grid(True)
plt.show()
