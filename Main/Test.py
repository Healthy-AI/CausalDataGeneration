from Algorithms.constrained_q_learning import ConstrainedQlearner
from Algorithms.greedyShuffledHistory import GreedyShuffled
from DataGenerator.data_generator import *
from Algorithms.q_learning import QLearner

seed = 12345
n_z = 6
n_x = 1
n_a = 7
n_y = 3
dist = DiscreteDistribution(n_z, n_x, n_a, n_y, seed=seed)
n_test_samples = 2000
test_data = generate_test_data(dist, n_test_samples)

n_actions = 3
n_outcomes = 3
n_x = 1
max_possible_outcome = 2
stop_action = 3
data = generate_data(DiscreteDistribution(3, n_x, n_actions, n_outcomes), 100)
data = split_patients(data)

cql = ConstrainedQlearner(1, n_actions, n_outcomes, data, learning_rate=0.01)
cql.learn()
cq = cql.q_table
print('''print(cq[0, -1, -1, -1])
print(cq[0, 0, -1, -1]),
print(cq[0, -1, -1, 2]),
print(cq[0, -1, -1, 0])''')
print(cq[0, -1, -1, -1])
print(cq[0, 0, -1, -1])
print(cq[0, -1, -1, 2])
print(cq[0, -1, -1, 0])

print('---------')
qlearner = QLearner(1, n_actions, n_outcomes, data, learning_rate=0.01)
q = qlearner.learn()

print(q[0, -1, -1, -1])
print(q[0, 0, -1, -1])
print(q[0, -1, -1, 2])
print(q[0, -1, -1, 0])
