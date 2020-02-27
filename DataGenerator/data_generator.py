import itertools
import json
from DataGenerator.distributions import *

import numpy as np


def generate_data(generator, n_samples):
    data = {'z': [], 'x': [], 'h': []}

    for i in range(n_samples):
        z = generator.draw_z()
        x = generator.draw_x(z)

        h = []

        for t in range(generator.n_treatments):
            a_t = generator.draw_a(h, x, z)
            y_t, done = generator.draw_y(a_t, h, x, z)
            h.append([a_t, y_t])
            if done:
                break
        data['z'].append(z)
        data['x'].append(x)
        data['h'].append(h)

    return data


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


data = generate_data(FredrikDistribution(), 300)
data = trim_data(data, 1)
data = split_patients(data)
write_json(data, "..\DataGeneratorTest\skewed_split_x")