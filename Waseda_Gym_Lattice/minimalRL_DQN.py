# this is a Deep Q Learning (DQN) agent including replay memory and a target network
# you can write a brief 8-10 line abstract detailing your submission and experiments here
# the code is based on https://github.com/seungeunrho/minimalRL/blob/master/dqn.py, which is released under the MIT licesne
# make sure you reference any code you have studied as above, with one comment line per reference

# imports
from collections import deque
# Note: deque is pronounced as “deck.” The name stands for double-ended queue.
import random
import pickle
# pytorch deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
gamma = 0.98    # discount rate
batch_size = 32
train_times = 10  # number of times train was run in a loop


class ReplayBuffer():
    """
    for DQN (off-policy RL), big buffer of experience
    you don't update weights of the NN as you run
    through the environment, instead you save
    your experience of the environment to this ReplayBuffer
    It has a max-size to fit in certain examples
    """
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            # a tuple that tells us what the state was
            # at a particular point in time
            # we store the current state, the action we chose,
            # the state we ended up in, and whether finished or not
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # converting the list to a single numpy.ndarray with numpy.array()
        # before converting to a tensor
        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_mask_lst = np.array(done_mask_lst)

        # print("torch.tensor(s_lst, dtype=torch.float) = ", torch.tensor(s_lst, dtype=torch.float))
        # print("torch.tensor(a_lst) = ", torch.tensor(a_lst))
        # print("torch.tensor(r_lst) = ", torch.tensor(r_lst))
        # print("torch.tensor(s_prime_lst, dtype=torch.float) = ", torch.tensor(s_prime_lst, dtype=torch.float))
        # print("torch.tensor(done_mask_lst) = ", torch.tensor(done_mask_lst))

        return torch.tensor(s_lst, device=device, dtype=torch.float), torch.tensor(a_lst, device=device), \
               torch.tensor(r_lst, device=device), torch.tensor(s_prime_lst, device=device, dtype=torch.float), \
               torch.tensor(done_mask_lst, device=device)

    def size(self):
        return len(self.buffer)

    def save(self, save_path):
        """save in .pkl file"""
        with open(save_path, 'wb') as handle:
            pickle.dump(self.buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self, file_path):
        """load a .pkl file"""
        with open(file_path, 'rb') as handle:
            self.buffer = pickle.load(handle)

class FCN_QNet(nn.Module):
    """
    action value function, Q(S, a)
    produce the actions in parallel as output vector,
    and choose the max
    """
    def __init__(self, insize, outsize):
        """
        insize ==> input size
            == size of the observation space
        outsize ==> output size
            == number of actions
        """
        super(FCN_QNet, self).__init__()
        self.fc1 = nn.Linear(insize, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, outsize)

    def forward(self, x):
        """
        standard 3-layer fully connected NN
        """
        x = x.to(device)  # for CUDA
        # print("input x.size() = ", x.size())
        x = x.view(x.size(0),-1)
        # may encounter view memory error
        # RuntimeError: view size is not compatible with input tensor's size and stride
        # (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        # x = x.reshape(x.size(0),-1)
        # print("after x.view ---> input x.size() = ", x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        """
        greedy epsilon choose
        """
        coin = random.random()
        if coin < epsilon:
            # print("coin < epsilon", coin, epsilon)
            # for 3actionStateEnv use [0,1,2]
            return random.randint(0,2)
        else:
            # print("exploit")
            out = self.forward(obs)
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    """
    core algorithm of Deep Q-learning

    do this training once per evaluation of the environment
    run evaluation once and train X times
    """
    for i in range(train_times):
        # sample from memory, which is not from the most recent runs
        # but from all previous runs in the memory, so you can be
        # more sample efficient, because you continuously learn from
        # past situations
        # key advantage of Off-policy
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        # the torch size is [batch_size, rows, cols], ie batch_first
        # print("DQN train --> s.size = ", s.size())
        # print(s)
        # print("DQN train --> r.size = ", r.size())
        # print(r)

        # forward once to q
        q_out = q(s)
        q_a = q_out.gather(1,a)
        # forward another time for q_target
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # calculate the target value
        # if environment is done, there is no future reward,
        # mask the final step reward with done_mask (0.0 if done else 1.0)
        target = r + gamma * max_q_prime * done_mask
        # L1 loss but smoothed out a bit
        loss = F.smooth_l1_loss(q_a, target)
        # we will try to improve on Q(s,a)
        # how well our Q-function is at guessing the future long-term rewards
        # Q_targ() is the target Q network, a 2nd NN to stablize training
        # Q(s,a) = R(s,a) + γ*Q_targ(s_prime)*done_mask
        optimizer.zero_grad()
        loss.backward()
        # clip the policy_net.parameters()
        # Dec05 2021 found it did not work...
        # for param in q.parameters():
        #     param.grad.data.clamp_(-1, 1)
        optimizer.step()
