from Algorithms.help_functions import hash_state
from Algorithms.statistical_approximator import StatisticalApproximator
from Algorithms.true_approximator import TrueApproximator
from DataGenerator.data_generator import generate_data, split_patients
from DataGenerator.distributions import DiscreteDistributionWithInformation
import numpy as np

n_z = 3
n_x = 1
n_a = 3
n_y = 3
dist = DiscreteDistributionWithInformation(n_z, n_x, n_a, n_y)
dist.print_detailed_treatment_statistics()
sa = StatisticalApproximator(n_x, n_a, n_y, split_patients(generate_data(dist, 5000)))
q = sa.generate_all_possible_histories([1, 3, 2])
print(q)
print(hash_state([0, 1], [-1, -1, -1, -1]))