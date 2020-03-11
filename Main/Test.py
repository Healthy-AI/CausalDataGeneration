from Algorithms.constrained_q_learning import ConstrainedQlearner
from Algorithms.greedyShuffledHistory import GreedyShuffled
from DataGenerator.data_generator import *
from Algorithms.q_learning import QLearner


n_actions = 3
n_outcomes = 3
max_possible_outcome = 2
stop_action = 3
data = generate_data(NewDistribution(), 1000)

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
