import time

from Algorithms.better_treatment_constraint import Constraint
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.constrained_greedy import ConstrainedGreedy
from Algorithms.exact_approximator import ExactApproximator
from Algorithms.statistical_approximator import StatisticalApproximator
from Algorithms.true_constraint import TrueConstraint
from DataGenerator.data_generator import split_patients, generate_data, generate_test_data
from DataGenerator.distributions import DiscreteDistributionWithSmoothOutcomes


def setup_data_sets(n_z, n_x, n_a, n_y, n_training_samples, n_test_samples, seed):
    start = time.time()
    print("Generating training and test data")
    dist = DiscreteDistributionWithSmoothOutcomes(n_z, n_x, n_a, n_y, seed=seed)
    training_data = split_patients(generate_data(dist, n_training_samples))
    test_data = generate_test_data(dist, n_test_samples)
    print("Generating data took {:.3f} seconds".format(time.time() - start))
    return dist, training_data, test_data


def setup_algorithms(training_data, dist, delta, train=True):
    start = time.time()
    n_x = dist.n_x
    n_a = dist.n_a
    n_y = dist.n_y
    statistical_approximation_prior = StatisticalApproximator(n_x, n_a, n_y, training_data, prior_mode='gaussian')
    true_approximation = ExactApproximator(dist)

    constraint_upper_stat = Constraint(training_data, n_a, n_y, approximator=statistical_approximation_prior, delta=delta)
    constraint_upper_true = Constraint(training_data, n_a, n_y, approximator=true_approximation, delta=delta)
    constraint_exact_true = TrueConstraint(dist, approximator=true_approximation, delta=delta)
    constraint_exat_stat = TrueConstraint(dist, approximator=statistical_approximation_prior, delta=delta)

    algorithms = [
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraint_upper_stat, statistical_approximation_prior, name="Dynamic Programming Upper Stat", label="CDP_US"),
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraint_upper_true, true_approximation, name="Dynamic Programming Upper True", label="CDP_UT"),
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraint_exact_true, true_approximation, name="Dynamic Programming Exact True", label="CDP_ET"),
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraint_exat_stat, statistical_approximation_prior, name="Dynamic Programming Exact Stat", label="CDP_ES"),
    ]

    print("Setting up algorithms took {:.3f} seconds".format(time.time() - start))
    return algorithms


def load_settings():
    starting_seed = 456102
    n_data_sets = 10
    n_deltas = 40
    n_z = 3
    n_x = 1
    n_a = 5
    n_y = 3
    n_training_samples = 7000
    n_test_samples = 2000
    file_name_prefix = "True_Approx_Sweep"

    return starting_seed, n_data_sets, n_deltas, n_z, n_x, n_a, n_y, n_training_samples, n_test_samples, file_name_prefix