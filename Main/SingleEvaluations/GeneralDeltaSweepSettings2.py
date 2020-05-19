import time

from Algorithms.better_treatment_constraint import Constraint
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.constrained_greedy import ConstrainedGreedy
from Algorithms.exact_approximator import ExactApproximator
from Algorithms.naive_dynamic_programming import NaiveDynamicProgramming
from Algorithms.naive_greedy import NaiveGreedy
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

    constraint_prior = Constraint(training_data, n_a, n_y, approximator=statistical_approximation_prior, delta=delta)
    constraint_prior = TrueConstraint(dist, approximator=statistical_approximation_prior, delta=delta)

    algorithms = [
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraint_prior, statistical_approximation_prior, name="Constrained Dynamic Programming", label="CDP"),
        ConstrainedGreedy(n_x, n_a, n_y, training_data, constraint_prior, statistical_approximation_prior, name="Constrained Greedy", label="CG"),
        NaiveGreedy(n_x, n_a, n_y, statistical_approximation_prior, round(delta * (n_a-1))+1, name='Naive Greedy', label='NG'),
        NaiveDynamicProgramming(n_x, n_a, n_y, training_data, statistical_approximation_prior, reward=-(delta+0.0001), name='Naive Dynamic Programming', label='NDP'),
    ]

    print("Setting up algorithms took {:.3f} seconds".format(time.time() - start))
    return algorithms


def load_settings():
    starting_seed = 10342
    n_data_sets = 10
    n_deltas = 40
    n_z = 3
    n_x = 1
    n_a = 5
    n_y = 3
    n_training_samples = 15000
    n_test_samples = 3000
    file_name_prefix = "GeneralComparisonDeltaSweep_15ksamples_"

    return starting_seed, n_data_sets, n_deltas, n_z, n_x, n_a, n_y, n_training_samples, n_test_samples, file_name_prefix