from Algorithms.help_functions import hash_state
from Algorithms.statistical_approximator import StatisticalApproximator
from Algorithms.true_approximator import TrueApproximator
from DataGenerator.data_generator import generate_data, split_patients
from DataGenerator.distributions import DiscreteDistributionWithInformation, DiscreteDistributionWithSmoothOutcomes
import numpy as np

seed = 103050  # Used for both synthetic and real data
n_z = 2
n_x = 1
n_a = 5
n_y = 3
dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed, outcome_sensitivity_x_z=1)
dist.print_treatment_statistics()
dist.print_detailed_treatment_statistics()

outcomes = np.zeros(2)
for i in range(100000):
    z = dist.draw_z()
    x = dist.draw_x(z)
    if np.max(x) == 0:
        y, done = dist.draw_y(1, None, x, z)
        outcomes[y] += 1
print(outcomes[0] / np.sum(outcomes))
sa = StatisticalApproximator(n_x, n_a, n_y, split_patients(generate_data(dist, 30000)))
