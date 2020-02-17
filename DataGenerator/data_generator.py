def generate_data(generator, n_samples):
    data = {'z': [], 'x': [], 'h': []}

    for i in range(n_samples):
        z = generator.draw_z()
        x = generator.draw_x(z)

        h = []

        for t in range(generator.n_treatments):
            a_t = generator.draw_a(h, x)
            y_t = generator.draw_y(a_t, h, x, z)
            h.append([a_t, y_t])
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