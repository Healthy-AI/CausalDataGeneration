from DataGenerator.data_generator import *

z = np.array([0.1, 0.2, 0.3, 0.4])
a00 = np.array([0.8, 0.1, 0.1, 0.1])
a12 = np.array([0.3, 0.3, 0.1, 0.2])
a22 = np.array([0.2, 0.1, 0.2, 0.1])

pa00 = np.sum(z*a00)

print(np.sum(z*a12))
print(np.sum (z*a22))

pzh = z*a00/pa00

print(np.sum(pzh*a12))
print(np.sum(pzh*a22))

def calc_score(pos):
    a00 = pos[0]
    a12 = pos[1]
    a22 = pos[2]
    pa00 = np.sum(z * a00)
    pzh = z*a00/pa00

    score = np.abs(np.sum(z*a12)-np.sum(z*a22))+np.abs(np.sum(pzh*a12)-np.sum(pzh*a22))
    if np.sign(np.sum(z*a12)-np.sum(z*a22)) == np.sign(np.sum(pzh*a12)-np.sum(pzh*a22)):
        score = 0
    if np.min(a00) < 0 or np.min(a12) < 0 or np.min(a22) < 0:
        score = 0
    if np.max(a00) > 1 or np.max(a12) > 1 or np.max(a22) > 1:
        score = 0
    return score

#Params
n_boids = 500
w = 0.5
wp = 0.4
wg = 0.3

limit = 0.01
boids_pos = np.zeros((500, 3, 4))
boids_best_pos = np.zeros((500, 3, 4))
boids_vel = np.zeros((500, 3, 4))
swarm_best_pos = np.random.uniform(limit, 1 - limit, (3, 4))
# Init boids
for i in range(n_boids):
    b_pos = boids_pos[i]
    b_vel = boids_vel[i]
    for j in range(3):
        for k in range(4):
            b_pos[j][k] = np.random.uniform(limit, 1 - limit)
            b_vel[j][k] = np.random.uniform(-(1 - limit), 1 - limit)
    boids_best_pos[i] = np.copy(b_pos)
    if calc_score(boids_best_pos[i]) > calc_score(swarm_best_pos):
        swarm_best_pos = np.copy(boids_best_pos[i])

for a in range(10):
    for i in range(n_boids):
        for j in range(3):
            for k in range(4):
                rp = np.random.uniform()
                rg = np.random.uniform()
                boids_vel[i][j][k] = w*boids_vel[i][j][k] + wp * rp * (boids_best_pos[i][j][k]-boids_pos[i][j][k]) + wg * rg * (swarm_best_pos[j][k] - boids_pos[i][j][k])
        boids_pos[i] += boids_vel[i]
        if calc_score(boids_pos[i]) > calc_score(boids_best_pos[i]):
            boids_best_pos[i] = np.copy(boids_pos[i])
            if calc_score(boids_best_pos[i]) > calc_score(swarm_best_pos):
                swarm_best_pos = np.copy(boids_best_pos[i])

    print(calc_score(swarm_best_pos))
    print(swarm_best_pos)
    a00 = swarm_best_pos[0]
    a12 = swarm_best_pos[1]
    a22 = swarm_best_pos[2]
    pa00 = np.sum(z * a00)
    print(np.sum(z * a12), np.sum(z * a22))
    pzh = z * a00 / pa00
    print(np.sum(pzh * a12), np.sum(pzh * a22))
