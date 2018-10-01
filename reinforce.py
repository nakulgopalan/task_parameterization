import argparse
import gym
import gym_param
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--mass', type=float, default=0.1, metavar='mass',
                    help='mass of pendulum, default, 0.1')
parser.add_argument('--length', type=float, default=0.5, metavar='length',
                    help='length of pendulum, default, 0.5')
args = parser.parse_args()


# env = gym.make('CartPole-v0')
env = gym.make('Cartpole-param-v0')
print(args.mass)
print(args.length)
# mass then length
env.param_switch(args.mass, args.length)
# env.seed(args.seed)
torch.manual_seed(args.seed)
hidden_layer_size = 4


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, hidden_layer_size)
        self.affine2 = nn.Linear(hidden_layer_size, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10

    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.9 + t * 0.1
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > 195.:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

    # model_parameters = policy.parameters()#filter(lambda p: p.requires_grad, policy.parameters())
    # print(model_parameters)

    vector = torch.nn.utils.parameters_to_vector(policy.parameters())
    print vector

    numpy_vector = vector.detach().numpy()
    name = 'weights_' + str(args.mass) + '_' + str(args.length) + '.txt'

    np.savetxt(name, numpy_vector, fmt='%f')

    # The next few lines load a policy model when needed!

    # new_vector = torch.from_numpy(numpy_vector)
    #
    # new_policy = Policy()
    # torch.nn.utils.vector_to_parameters(new_vector,new_policy.parameters())
    # print(torch.nn.utils.parameters_to_vector(new_policy.parameters()))




if __name__ == '__main__':
    main()