from Algorithms.better_treatment_constraint import Constraint
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.help_functions import hash_state
from Algorithms.statistical_approximator import StatisticalApproximator
from Algorithms.true_approximator import TrueApproximator
from DataGenerator.data_generator import generate_data, split_patients, generate_test_data
from DataGenerator.distributions import DiscreteDistributionWithInformation, DiscreteDistributionWithSmoothOutcomes
import numpy as np

t = []
t.append(np.load("cdp03_table.npy"))
t.append(np.load("cdp05_table.npy"))

seed = 3162  # Used for both synthetic and real data
n_z = 3
n_x = 1
n_a = 5
n_y = 3
n_training_samples = 100000
dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
dist.print_treatment_statistics()
dist.print_detailed_treatment_statistics()

split_training_data = split_patients(generate_data(dist, n_training_samples))

sa = StatisticalApproximator(n_x, n_a, n_y, split_training_data, prior_mode='gaussian')
print("Init constraints")
co03 = Constraint(n_x, n_a, n_y, approximator=sa, delta=0.3)
co05 = Constraint(n_x, n_a, n_y, approximator=sa, delta=0.5)

cdp03 = ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, co03, sa,
                              name="Dynamic Programming 0.3", label="CDP_03")
cdp05 = ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, co05, sa,
                              name="Dynamic Programming 0.5", label="CDP_05")

print("Training alg1")
cdp03.learn()
print("Trainging alg2")
cdp05.learn()

np.save("cdp03_table", cdp03.q_table)
np.save("cdp05_table", cdp05.q_table)

test_data = generate_test_data(dist, 50000)
tot03 = 0
tot05 = 0
for i in range(len(test_data)):
    e03 = cdp03.evaluate(test_data[i])
    e05 = cdp05.evaluate(test_data[i])
    if len(e03) < len(e05):
        print("len e03 < len e05")
        print(e03, "---", e05)
    bo03 = max([treatment[1] for treatment in e03])
    bo05 = max([treatment[1] for treatment in e05])
    if len(e03) > len(e05):
        print("len e03 > len e05")
        print(e03, "---", e05)
    tot03 += len(e03)
    tot05 += len(e05)

print(tot03 / len(test_data))
print(tot05 / len(test_data))

print("Hello world")