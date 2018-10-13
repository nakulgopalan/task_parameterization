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

policies_to_test = 10
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

print("time taken: " + total_n)

# plot
fig, ax = plt.subplots()

meanr = np.zeros(policies_to_test)
for i in range(policies_to_test):
    meanr[i] = results[i][0]

# the histogram of the data
ax.hist(meanr, 200)
ax.set_yscale('log')
ax.set_xlabel('Mean Steps')
ax.set_ylabel('# Random Policies')
ax.set_title('Mean Steps vs. Policies ({0})'.format(policies_to_test))

plt.savefig('{0}.png'.format(policies_to_test))
plt.show()
plt.close()
