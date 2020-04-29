import itertools

from Algorithms.better_treatment_constraint import Constraint
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.help_functions import hash_state
from Algorithms.statistical_approximator import StatisticalApproximator
from Algorithms.true_approximator import TrueApproximator
from DataGenerator.data_generator import generate_data, split_patients, generate_test_data
from DataGenerator.distributions import DiscreteDistributionWithInformation, DiscreteDistributionWithSmoothOutcomes
import numpy as np

seed = 78901  # Used for both synthetic and real data
n_z = 3
n_x = 1
n_a = 5
n_y = 3
n_training_samples = 500000
delta = 0.3
dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
dist.print_treatment_statistics()
dist.print_detailed_treatment_statistics()

split_training_data = split_patients(generate_data(dist, n_training_samples))

sa = StatisticalApproximator(n_x, n_a, n_y, split_training_data, prior_mode='gaussian')
ta = TrueApproximator(dist)
print("Init constraints")
csa = Constraint(n_x, n_a, n_y, approximator=sa, delta=delta)
cta = Constraint(n_x, n_a, n_y, approximator=ta, delta=delta)

cdp = ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, csa, sa,
                              name="Dynamic Programming", label="CDP")
cdpt = ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, cta, ta,
                              name="Dynamic Programming True", label="CDP_T")

print("Training alg1")
cdp.learn()
print("Trainging alg2")
cdpt.learn()

np.save("cdp_table", cdp.q_table)
np.save("cdpt_table", cdpt.q_table)

for x in range(2):
    for state in list(itertools.product(range(-1, n_y), repeat=n_a)):
        output = [0] * (n_a + 1)
        index = tuple(np.hstack((x, state)))
        if np.argmax(state) == 2:
            cdp.q_table[index] = output
            cdpt.q_table[index] = output
        cdp.q_table[index] = [sorted(cdp.q_table[index]*-1).index(x) for x in cdp.q_table[index]*-1]
        cdpt.q_table[index] = [sorted(cdpt.q_table[index]*-1).index(x) for x in cdpt.q_table[index]*-1]


new_q_table = cdp.q_table - cdpt.q_table
np.save("q_table_diff", new_q_table)
print("Hello world")