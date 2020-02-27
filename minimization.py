from scipy import optimize

def f(z):
    Egreedy = (z[0]+z[1])*1 + (z[2]+z[5])*2 + (z[3]+z[4])*3
    Esmart = (z[0]+z[3])*1 + (z[1]+z[2]+z[4]+z[5])*2
    return Esmart - Egreedy

cons = (
    {'type': 'ineq', 'fun': lambda z: z[0] - 0.05},
    {'type': 'ineq', 'fun': lambda z: z[1] - 0.05},
    {'type': 'ineq', 'fun': lambda z: z[2] - 0.05},
    {'type': 'ineq', 'fun': lambda z: z[3] - 0.05},
    {'type': 'ineq', 'fun': lambda z: z[4] - 0.05},
    {'type': 'ineq', 'fun': lambda z: z[5] - 0.05},
    {'type': 'ineq', 'fun': lambda z: z[4] - z[5]},
    {'type': 'ineq', 'fun': lambda z: z[0] + z[1] - z[0] - z[3]},   # E(A0) > E(A1)
    {'type': 'ineq', 'fun': lambda z: z[0] + z[3] - z[2] - z[5]},   # E(A1) > E(A2)
    {'type': 'ineq', 'fun': lambda z: z[0] - z[4]},
    {'type': 'ineq', 'fun': lambda z: z[2] + z[5] - z[3]},
    {'type': 'ineq', 'fun': lambda z: z[1] + z[2] - z[4]},
    {'type': 'ineq', 'fun': lambda z: z[2] + z[3] - z[4]},
    {'type': 'ineq', 'fun': lambda z: ((z[0] + z[1])*1 + (z[2] + z[5])*2 + (z[3] + z[4])*3) - ((z[0] + z[3])*1 + (z[1] + z[2] + z[4] + z[5])*2)},   # E(A1) < E(A0) long term
    {'type': 'ineq', 'fun': lambda z: ((z[2] + z[5])*1 + (z[0] + z[1])*2 + (z[3] + z[4])*3) - ((z[0] + z[3])*1 + (z[1] + z[2] + z[4] + z[5])*2)},   # E(A0) < E(A2) long term
    {'type': 'eq', 'fun': lambda z: z[0]+z[1]+z[2]+z[3]+z[4]+z[5]-1},
)

z0 = [0.35, 0.15, 0.15, 0.12, 0.2, 0.03]
res = optimize.minimize(f, z0, constraints=cons)
print(res)
print(f([0.3, 0.16, 0.16, 0.13, 0.2, 0.05]))
