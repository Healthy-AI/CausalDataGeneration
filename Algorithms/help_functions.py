def history_to_state(history, n_actions):
    state = [-1] * n_actions
    for intervention in history:
        treatment, outcome = intervention
        state[treatment] = outcome
    return state


def state_to_history(state):
    # Creates history from [0, 1, 2] to [[0, 0], [1, 1], [2, 2]]
    history = []
    for i, entry in enumerate(state):
        if entry != -1:
            history.append([i, entry])
    return history


def hash_history(x, history, n_actions):
    state = history_to_state(history, n_actions)
    string = hash_state(x, state)
    return string


def hash_state(x, state):
    xstring = ''.join(str(a) for a in x)
    statestring = ''.join(str(a) for a in state)
    string = xstring + statestring
    return string


def hash_array(x):
    xstring = ''.join(str(a) for a in x)
    return xstring
