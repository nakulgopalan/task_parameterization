# from a gist: https://gist.github.com/ts1829/d31611942df1f711d282107db0d3a764#file-policy-gradient-with-cartpole-and-pytorch-medium-version-ipynb

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
# %matplotlib inline

env = gym.make('CartPole-v1')
env.seed(1); torch.manual_seed(1);




#Hyperparameters
learning_rate = 0.01
gamma = 0.99
num_hidden = 128


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self.l1 = nn.Linear(self.state_space, num_hidden, bias=False)
        self.l2 = nn.Linear(num_hidden, self.action_space, bias=False)

        self.gamma = gamma

        self.softmax = nn.LogSoftmax(-1)

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor(), requires_grad=True)
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        x =  self.l1(x)
        x = self.l2(x)
            # nn.Dropout(p=0.6),
            # nn.ReLU(),
            # self.l2,
        x = self.softmax(x)

        return x

    # def forward(self, x):
    #     model = torch.nn.Sequential(
    #         self.l1,
    #         nn.Dropout(p=0.6),
    #         nn.ReLU(),
    #         self.l2,
    #         nn.Softmax(dim=-1)
    #     )
    #     return model(x)



policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def select_action(state):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    policy_nw = policy(Variable(state))
    c = Categorical(policy_nw)
    action = c.sample()
    # policy.policy_history.append(c.log_prob(action))

    policy.policy_history = torch.cat([policy.policy_history, Variable(torch.Tensor([c.log_prob(action)]), requires_grad=True) ])

    # # Add log probability of our chosen action to our history
    # print(policy.policy_history.nelement())
    # if policy.policy_history.nelement()!=0:
    #     # print(policy.policy_history.shape)
    #     # print(c.log_prob(action))
    #
    #     # policy.policy_history = torch.cat((policy.policy_history, c.log_prob(action)), 0)
    #     print(policy.policy_history)
    # else:
    #     policy.policy_history = policy.policy_history.append(c.log_prob(action))
    return action


def update_policy():
    R = 0
    rewards = []
    # policy.reward_episode = np.asarray(policy.reward_episode)
    # policy.reward_episode = (policy.reward_episode- policy.reward_episode.mean())/(policy.reward_episode.std() + 0.000000000000000001)

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 0.000000000000001)

    # Calculate loss
    # loss = Variable((torch.sum(torch.mul(policy.policy_history, Variable(rewards, requires_grad=True)).mul(-1), -1)), requires_grad=True)
    # # print(loss)
    # loss = loss/len(policy.reward_episode)
    loss = []
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        loss.append(-log_prob * reward)

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(policy.l2.cpu().weight)
    # print(policy.l1.cpu().weight)

    # Save and intialize episode history counters
    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []


def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset()  # Reset environment and record the starting state
        done = False

        for time in range(1000):
            action = select_action(state)

            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.data.cpu().numpy())

            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
            print(
            "Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
            break


# if __name__ == '__main__':
episodes = 1000
main(episodes)

window = int(episodes / 20)

fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9]);
rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
std = pd.Series(policy.reward_history).rolling(window).std()
ax1.plot(rolling_mean)
ax1.fill_between(range(len(policy.reward_history)), rolling_mean - std, rolling_mean + std, color='orange',
                 alpha=0.2)
ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
ax1.set_xlabel('Episode');
ax1.set_ylabel('Episode Length')

ax2.plot(policy.reward_history)
ax2.set_title('Episode Length')
ax2.set_xlabel('Episode');
ax2.set_ylabel('Episode Length')

fig.tight_layout(pad=2)
plt.show()
# fig.savefig('results.png')
