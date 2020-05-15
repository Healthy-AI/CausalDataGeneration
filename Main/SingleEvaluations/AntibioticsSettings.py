from Algorithms.naive_dynamic_programming import NaiveDynamicProgramming
from Algorithms.constrained_dynamic_programming import ConstrainedDynamicProgramming
from Algorithms.constrained_greedy import ConstrainedGreedy
from DataGenerator.data_generator import *
import time
from Algorithms.better_treatment_constraint import Constraint
from Algorithms.statistical_approximator import StatisticalApproximator
from Database.antibioticsdatabase import AntibioticsDatabase
from Algorithms.doctor import Doctor
from Algorithms.emulated_doctor import EmulatedDoctor
from Algorithms.doctor_approximator import DoctorApproximator


def setup_data_sets(seed):
    start = time.time()
    print("Generating training and test data")
    dist = AntibioticsDatabase(n_x=1, antibiotic_limit=50, seed=seed)
    training_data, test_data = dist.get_data()
    training_data = split_patients(training_data)

    print("Generating data took {:.3f} seconds".format(time.time() - start))
    return dist, training_data, test_data


def setup_algorithms(training_data, n_x, n_a, n_y, delta):
    start = time.time()
    statistical_approximation = StatisticalApproximator(n_x, n_a, n_y, training_data, prior_mode='gaussian')
    doctor_approximation = DoctorApproximator(n_x, n_a, n_y, training_data)

    print("Initializing Constraint")
    start = time.time()

    constraintStatUpper = Constraint(training_data, n_a, n_y, approximator=statistical_approximation, delta=delta, bound='upper')

    print("Initializing the constraint took {:.3f} seconds".format(time.time() - start))
    print("Initializing algorithms")
    algorithms = [
        ConstrainedGreedy(n_x, n_a, n_y, training_data, constraintStatUpper, statistical_approximation,
                          name='Constrained Greedy', label='CG'),
        # ConstrainedGreedy(n_x, n_a, n_y, split_training_data, constraintStatLower, statistical_approximation,
        #                   name='Constrained Greedy Lower', label='CG_L'),
        # ConstrainedGreedy(n_x, n_a, n_y, split_training_data, constraintFuncApprox, function_approximation,
        #                  name="Constrained Greedy FuncApprox", label="CG_F"),
        ConstrainedDynamicProgramming(n_x, n_a, n_y, training_data, constraintStatUpper,
                                      statistical_approximation),
        # ConstrainedDynamicProgramming(n_x, n_a, n_y, split_training_data, constraintStatUpper,
        #                              function_approximation, name="Constrained Dynamic Programming FuncApprox", label="CDP_F"),

        # NaiveGreedy(n_x, n_a, n_y, split_training_data),
        NaiveDynamicProgramming(n_x, n_a, n_y, training_data, statistical_approximation, reward=-0.35),
        Doctor(),
        EmulatedDoctor(n_x, n_a, n_y, training_data, approximator=doctor_approximation)
    ]
    return algorithms


def load_settings():
    starting_seed = 90821  # Used for both synthetic and real data
    delta = 0.0
    n_data_sets = 1
    file_name_prefix = 'antibioticsTest'
    return starting_seed, n_data_sets, delta, file_name_prefix