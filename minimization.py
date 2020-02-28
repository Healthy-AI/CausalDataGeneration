from scipy import optimize

def f(z):
    Egreedy = (z[0]+z[1])*1 + (z[2]+z[4]+z[5])*2 + (z[3]+z[6])*3
    Esmart = (z[0]+z[3])*1 + (z[1]+z[2]+z[4]+z[5]+z[6])*2
    return Esmart - Egreedy

cons = (
    {'type': 'ineq', 'fun': lambda z: z[0] - 0.01},
    {'type': 'ineq', 'fun': lambda z: z[1] - 0.01},
    {'type': 'ineq', 'fun': lambda z: z[2] - 0.01},
    {'type': 'ineq', 'fun': lambda z: z[3] - 0.01},
    {'type': 'ineq', 'fun': lambda z: z[4] - 0.01},
    {'type': 'ineq', 'fun': lambda z: z[5] - 0.01},
    {'type': 'ineq', 'fun': lambda z: z[6] - 0.01},
    {'type': 'eq', 'fun': lambda z: z[0]+z[1]+z[2]+z[3]+z[4]+z[5]+z[6]-1},
    {'type': 'ineq', 'fun': lambda z: z[0]+z[1]-z[0]-z[3]-z[6]},        # Greedy A0 > A1
    {'type': 'ineq', 'fun': lambda z: z[0]+z[3]+z[6]-z[2]-z[5]},        # Greedy A1 > A2
    {'type': 'ineq', 'fun': lambda z: z[4]-4*z[5]},                     # Choose to stop when ambigous Z4 & Z5
    {'type': 'ineq', 'fun': lambda z: z[1]-2.5*(z[4]+z[5])},            # When 1 on A1, worth it to try A0
    {'type': 'ineq', 'fun': lambda z: z[5]-(z[6])},                     # Z5 >= <6
    {'type': 'ineq', 'fun': lambda z: z[2]+z[5]-2.5*(z[3])},            # When 1 on A0, go to A3
    {'type': 'ineq', 'fun': lambda z: ((z[0]+z[1])*1 + (z[2]+z[4]+z[5]+z[6])*2 + z[3]*3) - ((z[0]+z[3]+z[6])*1 + (z[1]+z[2]+z[4]+z[5])*2) - 0.1},
    {'type': 'ineq', 'fun': lambda z: ((z[2]+z[5])*1 + (z[0]+z[1]+z[4]+z[6])*2 + z[3]*3) - ((z[0]+z[1])*1 + (z[2]+z[4]+z[5]+z[6])*2 + z[3]*3) - 0.1},
)

z0 = [0.3, 0.15, 0.35, 0.14, 0.04, 0.01, 0.01]
res = optimize.minimize(f, z0, constraints=cons)
print(res)

def tsts(z):
    print(((z[0]+z[1])*1 + (z[2]+z[4]+z[5]+z[6])*2 + z[3]*3))
    print(((z[0]+z[3]+z[6])*1 + (z[1]+z[2]+z[4]+z[5])*2))
    print(((z[2]+z[5])*1 + (z[0]+z[1]+z[4]+z[6])*2 + z[3]*3))
    print(sum(z))
tsts(z0)