import Algorithms.constrained_q_learning as cq_learning
import Algorithms.q_learning as q_learning
import Algorithms.greedyShuffledHistory as greedy
from DataGenerator.distributions import *
from DataGenerator.data_generator import *

n_actions = 3
n_outcomes = 3
max_possible_outcome = 2
stop_action = 3
greedyAlgorithm = greedy.GreedyShuffled(1, 3, n_actions)
data = generate_data(NewDistribution(), 3000)

data_dict = split_patients(data)
greedyAlgorithm.find_probabilities(data)

r = greedyAlgorithm.probabilities[-1, -1, -1]
print(r)