import numpy as np

times = np.load("saved_values/GeneralComparisonDeltaSweep_15ksamples_times.npy_old")
values = np.load("saved_values/GeneralComparisonDeltaSweep_15ksamples_values.npy_old")

print(times.shape)

print(times[:, :, 2])

times3 = np.zeros((10, 40))
values3 = np.zeros((10, 40))

for i in range(10):
    for j in range(40):
        times3[i, -(j+1)] = times[i, j, 2]
        values3[i, -(j+1)] = values[i, j, 2]

times[:, :, 2] = times3
values[:, :, 2] = values3

np.save("saved_values/GeneralComparisonDeltaSweep_15ksamples_times.npy_tmp", times)
np.save("saved_values/GeneralComparisonDeltaSweep_15ksamples_values.npy_tmp", values)