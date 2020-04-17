from Algorithms.true_approximator import TrueApproximator
from DataGenerator.data_generator import generate_data
from DataGenerator.distributions import DiscreteDistributionWithInformation
import numpy as np

dist = DiscreteDistributionWithInformation(3, 2, 3, 3)
approx = TrueApproximator(dist)
qq = approx.prepare_calculation(np.array([0, 1]), np.array([-1, 1, -1]), 0)
print(approx.calculate_probability(qq, 0))