import time

from Algorithms.better_treatment_constraint import Constraint
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.constrained_greedy import ConstrainedGreedy
from Algorithms.exact_approximator import ExactApproximator
from Algorithms.function_approximation import FunctionApproximation
from Algorithms.naive_dynamic_programming import NaiveDynamicProgramming
from Algorithms.naive_greedy import NaiveGreedy
from Algorithms.statistical_approximator import StatisticalApproximator
from DataGenerator.data_generator import split_patients, generate_data, generate_test_data
from Database.antibioticsdatabase import AntibioticsDatabase


def setup_data_sets(seed):
    start = time.time()
    print("Generating training and test data")
    dist = AntibioticsDatabase(n_x=n_x, antibiotic_limit=50, seed=seed)
    training_data, test_data = dist.get_data()
    training_data = split_patients(training_data)

    print("Generating data took {:.3f} seconds".format(time.time() - start))
    return dist, training_data, test_data


def setup_algorithms(training_data, dist, delta, train=True):
    start = time.time()
    n_x = dist.n_x
    n_a = dist.n_a
    n_y = dist.n_y
    statistical_approximation_prior = StatisticalApproximator(n_x, n_a, n_y, training_data, prior_mode='gaussian')
    function_approximation = FunctionApproximation(n_x, n_a, n_y, training_data)
    constraint_prior = Constraint(training_data, n_a, n_y, approximator=statistical_approximation_prior, delta=delta)

    constraintFuncApprox = Constraint(training_data, n_a, n_y, approximator=function_approximation, delta=delta)

    algorithms = [
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraint_prior, statistical_approximation_prior, name="Constrained Dynamic Programming", label="CDP"),
        ConstrainedGreedy(n_x, n_a, n_y, training_data, constraint_prior, statistical_approximation_prior, name="Constrained Greedy", label="CG"),
        ConstrainedGreedy(n_x, n_a, n_y, training_data, constraintFuncApprox, function_approximation,
                          name="Constrained Greedy FuncApprox", label="CG_F"),
        # ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraintStatUpper,
        #                              statistical_approximation),
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraintFuncApprox,
                                      function_approximation, name="Constrained Dynamic Programming FuncApprox",
                                      label="CDP_F"),

        NaiveDynamicProgramming(n_x, n_a, n_y, training_data, statistical_approximation_prior, reward=-(delta/2+0.0001), name='Naive Dynamic Programming', label='NDP'),
    ]

    print("Setting up algorithms took {:.3f} seconds".format(time.time() - start))
    return algorithms


def load_settings():
    starting_seed = 10342
    n_data_sets = 2
    n_deltas = 2
    file_name_prefix = "AntibioticsComparisonDeltaSweep_20deltasFixedSeedFA"

    return starting_seed, n_data_sets, n_deltas, file_name_prefix

n_x = 6