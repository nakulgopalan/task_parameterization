import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

def evaulate_policy(policy, env, number_of_episodes=2):
    """It evaluates a policy for number_of_episodes and returns and average score

    :param w: our policy is inner(w, s) > 0
    :type w: ndarray
    :param number_of_episodes: number of episodes we will run the policy for
    :type number_of_episodes: int
    :param env: environment object
    :type env: environment object
    :return: sum(timesteps_i)/number_of_episodes
    :rtype: float
    """

    results = []
    # for e in range(number_of_episodes):
    s_old = env.reset()
    t = 0
    done = False
    while not done:
        # Choose action
        action = None
        if policy(torch.FloatTensor(s_old)).item() > 0:
            action = 1
        else:
            action = 0
        # Take action
        s_new, r, done, _ = env.step(action)

        # Update
        s_old = s_new

        t += 1
        if t>200:
            break
        # results.append(t)
        # print(t)
    return t
