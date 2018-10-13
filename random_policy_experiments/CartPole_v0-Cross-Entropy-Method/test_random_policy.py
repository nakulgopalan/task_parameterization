""" Solving the CartPole-v0 environment with the Cross Entropy Method """

__author__ = "Jan Krepl"
__email__ = "jankrepl@yahoo.com"
__license__ = "MIT"

import gym
from foo import *
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt

# PARAMETERS
t0 = time.time()


policies_to_test = 1000
episodes_for_evaluation = 100

env = gym.make('CartPole-v0')

counter = 0

long_np = None

results = {}
# MAIN ALGORITHM
for i in range(policies_to_test):
    print(i)
    w = np.random.rand(4)
    eps = 0
    rsum = 0.0
    failed = False
    for j in range(1, episodes_for_evaluation + 1):
        r = evaulate_policy(w, env, number_of_episodes=1)
        rsum += r
        if r >= 195 and not failed:
            eps = j
        else:
            failed = True
    results[i] = (eps, rsum / episodes_for_evaluation, w)

env.close()

t1 = time.time()

total_n = t1-t0
print(total_n)

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

eplen = np.zeros(policies_to_test)
meanr = np.zeros(policies_to_test)
tosave = np.zeros((policies_to_test, 6))
for i in range(policies_to_test):
    eplen[i] = results[i][0]
    meanr[i] = results[i][1]
    tosave[i, :] = np.concatenate((np.array([results[i][0], results[i][1]], dtype=np.float), results[i][2]))

np.save('{0}'.format(policies_to_test), tosave)
print(tosave)

# the histogram of the data
ax1.hist(eplen, episodes_for_evaluation)
ax1.set_yscale('log')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('# Random Policies')
ax1.set_title('Number of Random Episodes Completed vs. Policies ({0})'.format(policies_to_test))

ax2.hist(meanr, 200)
ax2.set_yscale('log')
ax2.set_xlabel('Mean Reward')
ax2.set_ylabel('# Random Policies')
ax2.set_title('Mean Reward vs. Policies ({0})'.format(policies_to_test))

plt.savefig('{0}.png'.format(policies_to_test))
plt.show()
plt.close()
