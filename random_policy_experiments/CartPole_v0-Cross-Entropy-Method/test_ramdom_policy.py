""" Solving the CartPole-v0 environment with the Cross Entropy Method """

__author__ = "Jan Krepl"
__email__ = "jankrepl@yahoo.com"
__license__ = "MIT"

import gym
from foo import *
import numpy as np
import time

# PARAMETERS
t0 = time.time()

# init_mu = np.random.rand(4)#[0, 0, 0, 0]
policies_to_test = 100000

episodes_for_evaluation = 1  # each policy run for episodes_for_evaluation episodes and then result is the mean score
n_samples = 1  # number of sampled weights we generate
n_best_to_keep = 5  # We order policies based on results and then use n_best_to_keep best of them to estimate new param
absolute_winners_threshold = 200  # algorithm terminated when a certain number of winning policies found
initial_noise_coef = 1  # we always add constant_noise_coef * I to the estimated covmat (to increase variance)
noise_decay = 99 / 100

# INITIALIZATIO
env = gym.make('CartPole-v0')

# global_max = 0
# absolute_winners = []  # list of policies(w) that achieved perfect score of 199
# mu = init_mu
# covmat = init_covmat
# noise_coef = initial_noise_coef

counter = 0

long_np = None

# MAIN ALGORITHM
for i in range(policies_to_test):
    print(i)
    w = np.random.rand(4)
    samples_result = evaulate_policy(w, env, number_of_episodes=1)
    # print(w)
    # print(samples_result)
    v = np.append(w,float(samples_result)).reshape(1,5)
    if i==0:
        long_np = v
    else:
        long_np = np.concatenate((long_np,v))


np.savetxt('results.txt',long_np,fmt='%f')

env.close()

t1 = time.time()

total_n = t1-t0
print(total_n)
print(long_np)