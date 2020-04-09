from Algorithms.betterTreatmentConstraint import Constraint
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.online_q_learning import OnlineQLearner
from Algorithms.q_learning_with_constraint import QLearnerConstrained
from DataGenerator.data_generator import generate_test_data, split_patients, generate_data
from DataGenerator.distributions import DiscreteDistributionWithSmoothOutcomes, NewDistribution
import numpy as np
import matplotlib.pyplot as plt

n_runs = 6

seed = None
n_z = 3
n_x = 1
n_a = 3
n_y = 3
training_episodes = 100000
n_training_samples = 2000
n_test_samples = 2000
delta = 0.25
epsilon = 0
prior_power = 1.96

plot_colors = ['k', 'r', 'b', 'g', 'm', 'c', 'y']
plot_markers = ['', '--', ':']

dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=n_z / n_x)
#dist = NewDistribution()
training_data = generate_data(dist, n_training_samples)
split_training_data = split_patients(training_data)
test_data = generate_test_data(dist, n_test_samples)
print("Generated Data")
constraint = Constraint(split_training_data, n_a, n_y, delta=delta, epsilon=epsilon, prior_power=prior_power)
#dist.print_moderator_statistics()
#dist.print_covariate_statistics()
#dist.print_treatment_statistics()
evaluations = {}
pp = [0.00001, 1.96, 4, 10, 1000, 100000]

for i in range(n_runs):
    #oql = OnlineQLearner(n_x, n_a, n_y, dist, constraint, learning_time=20000 + i * 20000)
    constraint = Constraint(split_training_data, n_a, n_y, delta=delta, epsilon=epsilon, prior_power=pp[i])
    oql = ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraint, prior_power=pp[i])
    oql.learn()

    alg_evals = []
    for j in range(n_test_samples):
        alg_evals.append(oql.evaluate(test_data[j]))
    evaluations[i] = alg_evals
    print("Done with run {}".format(i))


# Calculate max mean treatment effect over population
max_mean_treatment_effects = np.zeros((n_runs, n_a + 1))
for i in range(n_runs):
    for i_sample in range(n_test_samples):
        treatments = evaluations[i][i_sample]
        best_found = 0
        for i_treatment in range(len(max_mean_treatment_effects[i])):
            if i_treatment < len(treatments):
                effect = treatments[i_treatment][1]
                if effect > best_found:
                    best_found = effect
            max_mean_treatment_effects[i][i_treatment] += best_found
max_mean_treatment_effects /= n_test_samples

# Calculate mean treatment effect over population
mean_treatment_effects = np.zeros((n_runs, n_a + 1))      # Overshoot by 1 to get all max values at last step
mean_num_tests = np.zeros(n_runs)

for i_sample in range(n_test_samples):
    for i in range(n_runs):
        treatments = evaluations[i][i_sample]
        mean_num_tests[i] += len(treatments)
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
for i in range(n_runs):
    plt.plot(x, mean_treatment_effects[i], plot_colors[i] + plot_markers[0], label="g={}".format(pp[i]))
    plt.plot(x, max_mean_treatment_effects[i], plot_colors[i] + plot_markers[1])
    plt.fill_between(x, mean_treatment_effects[i], max_mean_treatment_effects[i], color=plot_colors[i], alpha=0.1)
    plt.axvline(mean_num_tests[i]-1, 0, average_max_treatment_effect, color=plot_colors[i])

plt.grid(True)
plt.xticks(x, x_ticks)
plt.plot(x, np.ones(len(x))*average_max_treatment_effect, label='MAX_POSS_AVG')

plt.legend(loc='lower right')
plt.show(block=False)