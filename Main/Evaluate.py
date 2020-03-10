from Algorithms.constrained_q_learning import ConstrainedQlearner
from Algorithms.q_learning import QLearner
from Algorithms.greedyShuffledHistory import GreedyShuffled
from DataGenerator.distributions import *
import Algorithms.greedyShuffledHistory as greedy
import matplotlib.pyplot as plt
from DataGenerator.data_generator import *
import Algorithms.q_learning as ql
import time



seed = 0
n_z = 3
n_x = 1
n_a = 5
n_y = 3
n_algorithms = 1
n_training_samples = 1000
n_test_samples = 100
main_start = time.time()
plot_colors = ['k', 'r--', 'b:', 'g-.']

# Generate the data
dist = DiscreteDistribution(n_z, n_x, n_a, n_y, seed=seed)
start = time.time()
print("Generating {} training samples...".format(n_training_samples))
training_data = generate_data(dist, n_training_samples)
split_training_data = split_patients(training_data)
print("Generating training samples took {:.3f} seconds".format(time.time()-start))
start = time.time()
print("Generating {} test samples...".format(n_test_samples))
test_data = generate_test_data(dist, n_test_samples)
print("Generating test samples took {:.3f} seconds".format(time.time()-start))


# Initialize and train the algorithms
# TODO Init Greedy
# start = time.time()
# print("Initializing Greedy...")
# G = greedy.GreedyShuffled(n_x, n_y, n_a)
# print("\tTraining Greedy...")
# G.find_probabilities(training_data)
# print("\tTraining the Greedy algorithm took {:.3f} seconds".format(time.time()-start))
# TODO Init Constrained Q Learning
# start = time.time()
# print("Initializing Constrained Q-learning...")
# CQL = cql.ConstrainedQlearner(n_x, n_y, n_a, split_training_data, learning_rate=0.01, discount_factor=1)
# print("\tTraining Constrained Q-learning...")
# CQL.learn()
# print("\tTraining the Constrained Q-learning algorithm took {:.3f} seconds".format(time.time()-start))
start = time.time()
print("Initializing Q-learning...")
QL = QLearner(n_x, n_a, n_y, split_training_data, reward=-0.1, learning_rate=0.01, discount_factor=1)
print("\tTraining Q-learning...")
QL.learn()
print("\tTraining the Q-learning algorithm took {:.3f} seconds".format(time.time()-start))

# Evaluate the algorithms
evaluations = [[]]
for i in range(n_test_samples):
    # evaluations[0].append(
    # evaluations[1].append(CQL.evaluate(test_data[i]))
    e = QL.evaluate(test_data[i])
    print(e)
    evaluations[0].append(QL.evaluate(test_data[i]))

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
            if effect < best_found:
                print(i_treatment)
            if effect > best_found:
                best_found = effect
            mean_treatment_effects[i_alg][i_treatment] += effect
mean_treatment_effects /= n_test_samples
# Plot mean treatment effect over population
x = np.arange(0, n_a+1)
for i_plot in range(n_algorithms):
    plt.plot(x, mean_treatment_effects[0], plot_colors[i_plot])
plt.xlabel('Treatment index')
plt.ylabel('Mean treatment effect over population')
plt.show()

# Calculate max mean treatment effect over population
max_mean_treatment_effects = np.zeros((n_algorithms, n_a))
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
# Plot mean treatment effect over population
x = np.arange(0, n_a)
for i_plot in range(n_algorithms):
    plt.plot(x, max_mean_treatment_effects[0], plot_colors[i_plot])
plt.xlabel('Treatment index')
plt.ylabel('Max mean treatment effect over population')
plt.show()



print("Running Evaluate took {:.3f} seconds".format(time.time()-main_start))

