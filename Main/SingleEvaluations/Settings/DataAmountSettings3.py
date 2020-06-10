import time

from Algorithms.Constraints.better_treatment_constraint import Constraint
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.constrained_greedy import ConstrainedGreedy
from Algorithms.Approximators.exact_approximator import ExactApproximator
from Algorithms.Approximators.statistical_approximator import StatisticalApproximator
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


def setup_algorithms(training_data, dist, delta):
    start = time.time()
    n_x = dist.n_x
    n_a = dist.n_a
    n_y = dist.n_y
    statistical_approximation_prior = StatisticalApproximator(n_x, n_a, n_y, training_data, smoothing_mode='gaussian')
    statistical_approximation_none = StatisticalApproximator(n_x, n_a, n_y, training_data, smoothing_mode='none')
    true_approximation = ExactApproximator(dist)
    constraint_prior = Constraint(training_data, n_a, n_y, approximator=statistical_approximation_prior, delta=delta)
    constraint_none = Constraint(training_data, n_a, n_y, approximator=statistical_approximation_none, delta=delta)
    constraint_true = Constraint(training_data, n_a, n_y, approximator=true_approximation, delta=delta)
    algorithms = [
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraint_true, true_approximation, name="Dynamic Programming True", label="CDP_T"),
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraint_prior, statistical_approximation_prior, name="Constrained Dynamic Programming", label="CDP_H"),
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraint_none, statistical_approximation_none, name="Constrained Dynamic Programming Uninformed", label="CDP_U"),
        ConstrainedGreedy(n_x, n_a, n_y, training_data, constraint_prior, statistical_approximation_prior, name="Constrained Greedy", label="CG"),
        #NaiveGreedy(n_x, n_a, n_y, statistical_approximation_prior, round(delta * (n_a-1))+1, name='Naive Greedy', label='NG'),
        #NaiveDynamicProgramming(n_x, n_a, n_y, training_data, statistical_approximation_prior, reward=-(delta+0.0001), name='Naive Dynamic Programming', label='NDP'),
    ]

    print("Setting up algosrithms took {:.3f} seconds".format(time.time() - start))
    return algorithms


def load_settings():
    starting_seed = 476891
    n_data_sets = 10
    n_data_set_sizes = 30
    delta = 0.3
    n_z = 3
    n_x = 1
    n_a = 5
    n_y = 3
    n_training_samples_max = 100000
    n_test_samples = 3000
    file_name_prefix = "DataSweep_50k2_"

    return starting_seed, n_data_sets, delta, n_data_set_sizes, n_z, n_x, n_a, n_y, n_training_samples_max, n_test_samples, file_name_prefix