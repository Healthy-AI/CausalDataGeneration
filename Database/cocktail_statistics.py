from Database.sql_cocktail_statistics import get_antibiotcsevents
from Database.antibioticsdatabase import AntibioticsDatabase
import matplotlib.pyplot as plt
import numpy as np
from Database.treatment_to_test import treatment_to_test

database = AntibioticsDatabase()
database.cur.execute(get_antibiotcsevents)
data = database.cur.fetchall()
datapoints = {}
prev_hadm_id = 0
for chartevent in data:
    hadm_id = chartevent[0]
    label = chartevent[1]
    start_time = chartevent[2]
    if hadm_id in datapoints:
        datapoints[hadm_id].append([label, start_time])
    else:
        datapoints[hadm_id] = [[label, start_time]]

times = []
for hadm_id, value in datapoints.items():
    for i, entry in enumerate(value):
        label = entry[0]
        time = entry[1]
        if i != 0:
            if label != prev_label:
                diff_time = time - prev_time
                minutes_diff_time = diff_time.seconds/60
                times.append(minutes_diff_time)
        prev_label = label
        prev_time = time

plt.hist(times, bins=int(1400/60))
plt.xticks(np.arange(0, 1441, 60), rotation='vertical')
plt.xlabel('Minutes between different antibiotics')
plt.ylabel('Quantity')
plt.show()
n_zeros = len(times) - np.count_nonzero(times)
print(n_zeros, 'zeros out of', len(times), 'which is', n_zeros/len(times), '% of the switching cases')

used_data = database.get_data()
used_times = []
for hadm_id in used_data[0]['z']:
    if hadm_id in datapoints:
        value = datapoints[hadm_id]
        for i, entry in enumerate(value):
            label = entry[0]
            time = entry[1]
            if i != 0:
                if label != prev_label:
                    diff_time = time - prev_time
                    minutes_diff_time = diff_time.seconds / 60
                    used_times.append(minutes_diff_time)
            prev_label = label
            prev_time = time



plt.hist(used_times, bins=int(1400/60))
plt.xticks(np.arange(0, 1441, 60), rotation='vertical')
plt.xlabel('Minutes between different antibiotics')
plt.ylabel('Quantity')
plt.show()
n_zeros = len(used_times) - np.count_nonzero(used_times)
print(n_zeros, 'zeros out of', len(used_times), 'which is', n_zeros/len(used_times), '% of the switching cases')
