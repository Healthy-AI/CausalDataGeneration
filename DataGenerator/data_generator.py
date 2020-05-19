import itertools
import json
from multiprocessing.pool import Pool
import numpy as np


def generate_sample(generator):
    z = generator.draw_z()
    x = generator.draw_x(z)

    h = []

    for t in range(generator.n_a):
        a_t = generator.draw_a(h, x, z)
        y_t, done = generator.draw_y(a_t, h, x, z)
        h.append(np.array([a_t, y_t]))
        if done:
            break
    return z, x, h


def generate_data(generator, n_samples):
    data = {'z': [], 'x': [], 'h': []}

    results = []
    for i in range(n_samples):
        results.append(generate_sample(generator))

    for i in range(n_samples):
        z, x, h = results[i]
        data['z'].append(z)
        data['x'].append(x)
        data['h'].append(h)

    return data


# Generates counterfactual data with all results for all treatments
def generate_test_data(generator, n_samples):
    data = []
    for i in range(n_samples):
        z = generator.draw_z()
        x = generator.draw_x(z)
        subject = []
        subject.append(z)
        subject.append(x)
        subject.append(np.zeros(generator.n_a))
        h = []
        for a in range(generator.n_a):
            y, _ = generator.draw_y(a, h, x, z)
            subject[2][a] = y
            h.append(np.array([a, y]))
        data.append(np.array(subject))
    return np.array(data)


def trim_data(data, threshold):
    trimmed_data = data.copy()
    for i in range(len(trimmed_data['z'])):
        tmp_h = trimmed_data['h'][i]
        for j in range(len(tmp_h)):
            if tmp_h[j][1] >= threshold:
                trimmed_data['h'][i] = tmp_h[0:j+1]
                break

    return trimmed_data


def write_json(data, name):
    with open(name + '.json', 'w') as f:
        json.dump(data, f, default=convert)


def read_json(name):
    with open(name + '.json') as f:
        data = json.load(f)
    return data


def convert(o):
    if isinstance(o, np.int64):
        return int(o)


def split_patients(data):
    x = data['x']
    h = data['h']
    z = data['z']
    new_x = []
    new_h = []
    new_z = []
    for i, history in enumerate(h):
        for j in range(0, len(history)):
            new_x.append(x[i])
            new_z.append(z[i])
            new_h.append(h[i][0:len(h[i])-j])

    data['x'] = new_x
    data['h'] = new_h
    data['z'] = new_z
    return data