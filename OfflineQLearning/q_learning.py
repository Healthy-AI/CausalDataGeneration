import numpy as np

n_actions = 4
n_results = 3
n_x = 2
learning_rate = 0.2
discount_factor = 0.9

# Q-table indexed with x, y_0, y_1, y_2, y_3 and a
q_table = np.zeros((n_x,) + (n_results + 1,) * n_actions + (n_actions + 1,))


def learn(history):
    for _ in range(10000):
        state, action, reward, next_state = history[np.random.randint(0, len(history))]

        if action == 0:
            q_table[state, action] = reward
        else:
            q_table[state, action] = q_table[state, action] + learning_rate \
                                 * (reward + discount_factor * max(q_table[next_state]) - q_table[state, action])
    print(q_table)

def transform_data():
    pass
