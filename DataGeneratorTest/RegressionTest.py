from sklearn import linear_model
from data_generator import *
from distributions import SimpleDistribution
import numpy as np


#data = generate_data(SimpleDistribution(), 100)
#data = trim_data(data, 2)

data = read_json("simple")
x = []
t = []
y = []

for i in range(len(data['x'])):
    for j in range(len(data['h'][i])):
        x.append(data['x'][i])
        t.append(data['h'][i][j][0])
        y.append(data['h'][i][j][1])

x = np.array(x)
y = np.array(y)

t0 = np.array(t) == 0
t1 = np.array(t) == 1
t2 = np.array(t) == 2
t3 = np.array(t) == 3

t_array = [t0, t1, t2, t3]
regressors = []

for i in range(4):
    lin = linear_model.LinearRegression()
    model = lin.fit(x[t_array[i]].reshape(-1, 1), y[t_array[i]])
    regressors.append(lin)

print(regressors[0].predict([[0]]))
print(regressors[1].predict([[0]]))
print(regressors[2].predict([[0]]))
print(regressors[3].predict([[0]]))
