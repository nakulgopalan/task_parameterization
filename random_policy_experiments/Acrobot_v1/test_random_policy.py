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

policies_to_test = 10000
episodes_for_evaluation = 100

env = gym.make('Acrobot-v1')

counter = 0

long_np = None

results = {}
# MAIN ALGORITHM
for i in range(policies_to_test):
    print(i)
    w = np.random.rand(6)
    rsum = 0.0
    min_steps = float('+inf')
    for j in range(1, episodes_for_evaluation + 1):
        r = evaulate_policy(w, env, number_of_episodes=1)
        rsum += r
        if (min_steps > r):
            min_steps = r

    results[i] = (rsum / episodes_for_evaluation, min_steps,w)
    print(results[i])

env.close()

t1 = time.time()

total_n = t1-t0

print("time taken: " + str(total_n))



meanr = np.zeros(policies_to_test)
minlen = np.zeros(policies_to_test)
tosave = np.zeros((policies_to_test, 8))
for i in range(policies_to_test):
    meanr[i] = results[i][0]
    minlen[i] = results[i][1]
    tosave[i, :] = np.concatenate((np.array([results[i][0], results[i][1]], dtype=np.float), results[i][2]))


# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
# the histogram of the data
ax1.hist(minlen, episodes_for_evaluation)
ax1.set_yscale('log')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('# Random Policies')
# ax1.set_title('Histogram of Minimum Length from Random Policies on the Acrobot Gym Environment vs. Policies ({0})'.format(policies_to_test))

ax2.hist(meanr, 200)
ax2.set_yscale('log')
ax2.set_xlabel('Mean Reward')
ax2.set_ylabel('# Random Policies')
# ax2.set_title('Histogram of Mean Reward from Random policies on the Acrobot Gym Environment vs. Policies ({0})'.format(policies_to_test))

plt.savefig('{0}.png'.format(policies_to_test))
plt.show()
plt.close()