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

        q_table[state, action] = q_table[state, action] + learning_rate \
                                 * (reward + discount_factor * max(q_table[next_state]) - q_table[state, action])
        stop_reward = -np.infty
        if max(next_state[1:]) > 0:
            stop_reward = max(next_state[1:])
        q_table[next_state, -1] = stop_reward


    print(q_table)


def transform_data():
    pass


def convert_to_sars(data, n_actions):
    x = data['x']
    h = data['h']
    all_sars = []
    for i, patient in enumerate(x):
        actions = [-1] * n_actions
        outcomes = [-1] * n_actions
        for treatment in h[i]:
            action, outcome = treatment
            actions[action] = outcome
            outcomes[action] = outcome

        for j in range(len(h[i])):
            if h[i][j] != -1:
                temp_actions = actions.copy()
                new_action = h[i][j][0]
                temp_actions[new_action] = -1
                s = [patient, temp_actions]
                a = new_action + 1
                r = reward(h[i])
                new_actions = temp_actions.copy()
                new_actions[new_action] = 1
                s_prime = [patient, new_actions]
                sars = (s, a, r, s_prime)
                all_sars.append(sars)
    return all_sars


def reward(history):
    maxy = max(list(h[1] for h in history))
    if maxy > 0:
        r = maxy - 0.5 * len(history)
    else:
        r = -np.Inf
    return r
