from Algorithms.constrained_q_learning import ConstrainedQlearner
from Algorithms.q_learning import QLearner, convert_to_sars
from Algorithms.greedyShuffledHistory import GreedyShuffled
from DataGenerator.distributions import *
from DataGenerator.data_generator import *

n_actions = 3
n_outcomes = 3
max_possible_outcome = 2
stop_action = 3
greedyAlgorithm = GreedyShuffled(1, 3, n_actions)
data = generate_data(NewDistribution(), 500)

data = split_patients(data)
#greedyAlgorithm.find_probabilities(data)
cql = ConstrainedQlearner(1, n_outcomes, n_actions, data, learning_rate=0.01)
cql.fill_table()
cq = cql.q_table

print(cq[0, -1, -1, -1])
print(cq[0, 1, -1, -1])
print(cq[0, 1, 2, -1])
print(cq[0, 2, -1, -1])
print(cq[0, 1, 1, -1])
print(cq[0, -1, 1, 1])

print('---------')
ql = QLearner(1, n_outcomes, n_actions, learning_rate=0.01)
q = ql.learn(convert_to_sars(data, n_actions))

print(q[0, -1, -1, -1])
print(q[0, 1, -1, -1])
print(q[0, 1, 2, -1])
print(q[0, 2, -1, -1])
print(q[0, 1, 1, -1])
print(q[0, -1, 1, 1])

