import numpy as np
# we sample a 10,000 task parameters for mass and length mass in interval (0.1 and 10.) kgs and length in interval
np.random.seed(seed=10)
m0 = 0.1
m1 = 10.
l0 = .5
l1 = 10.

m = (m1-m0)*np.random.random(10000)+m0
l = (l1-l0)*np.random.random(10000)+l0

tau = np.stack((m, l), axis=1)
print(tau.shape)
print(tau[1][1])
np.savetxt('tau.txt', tau, fmt='%f')
load_tau = np.loadtxt('tau.txt', dtype=float)
print(load_tau)