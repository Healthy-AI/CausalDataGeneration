import numpy as np

n_actions = 4
n_results = 3
n_x = 2
learning_rate = 0.2
discount_factor = 1

# Q-table indexed with x, y_0, y_1, y_2, y_3 and a
q_table = np.zeros((n_x,) + (n_results + 1,) * n_actions + (n_actions + 1,))

# Initialize all final states with the rewards for picking that state
for x in range(len(q_table)):
    for y_0 in range(-1, len(q_table[x]) - 1):
        for y_1 in range(-1, len(q_table[x][y_0]) - 1):
            for y_2 in range(-1, len(q_table[x][y_0][y_1]) - 1):
                for y_3 in range(-1, len(q_table[x][y_0][y_1][y_2]) - 1):
                    q_table[x][y_0][y_1][y_2][y_3][-1] = max(max(y_0, y_1), max(y_2, y_3))
                    if q_table[x][y_0][y_1][y_2][y_3][-1] < 1:
                        q_table[x][y_0][y_1][y_2][y_3][-1] = -np.infty


def to_index(state):
    return state[0], state[1][0], state[1][1], state[1][2], state[1][3]


def learn(history):
    for _ in range(10000):
        state, action, reward, next_state = history[np.random.randint(0, len(history))]

        q_table[to_index(state) + (action,)] = q_table[to_index(state) + (action,)] + learning_rate \
                                               * (reward + discount_factor * max(q_table[to_index(next_state)])
                                                  - q_table[to_index(state) + (action,)])


learn([[[[0], [-1, -1, -1, -1]], 1, -0.5, [[0], [-1, 2, -1, -1]]]])
