"""
perform inference based on trained model weights
Oct 16 2022
"""
import gym
import random
import numpy as np
# pytorch deep learning
from minimalRL_DQN import (
    ReplayBuffer,
    FCN_QNet,
    train,
    device,
    gamma,
    batch_size,
    train_times,
)
from RNN_pytorch import (
    RNN_LSTM_onlyLastHidden,
    BRNN,
)
from count_param_pytorch import count_parameters

import torch
import torch.nn.functional as F
import torch.optim as optim

import os # for creating directories
import sys
import datetime

# time the program
from time import time
from timer import secondsToStr, time_log

N_mer = "48mer"

if N_mer == "20mer-B":
    seq = "hhhpphphphpphphphpph"  # 20mer-B
    q_state_dict_path = "./Sample_DQN_weights/256L2/20merB-example-state_dict.pth"
elif N_mer == "36mer":
    seq = "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP"  # 36mer
    q_state_dict_path = "./Sample_DQN_weights/256L2/36mer-example-state_dict.pth"
elif N_mer == "48mer":
    seq = "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH"  # 48mer
    # 2-layer LSTM architecture (in 256L2 dir)
    q_state_dict_path = "./Sample_DQN_weights/256L2/48mer-256L2-example-state_dict.pth"
    # 3-layer LSTM architecture (in 512L3 dir)
    q_state_dict_path = "./Sample_DQN_weights/512L3/48mer-512L3-example-state_dict.pth"


seq = seq.upper()  # Require HP upper case!
num_episodes = 1
# @hyperparameters
max_steps_per_episode = len(seq)

# Nov30 2021 add one more column of step_E
hp_depth = 2  # {H,P} binary alphabet
action_depth = 4  # 0,1,2,3 in observation_box
energy_depth = 0  # state_E and step_E
# one hot the HP seq
seq_bin_arr = np.asarray([1 if x == 'H' else 0 for x in seq])
seq_one_hot = F.one_hot(torch.from_numpy(seq_bin_arr), num_classes=hp_depth)
seq_one_hot = seq_one_hot.numpy()
# print(f"seq({seq})'s one_hot = ")
# print(seq_one_hot)
init_HP_len = 2  # initial two HP units placed
first_two_actions = np.zeros((init_HP_len,), dtype=int)

def one_hot_state(state_arr, seq_one_hot, action_depth):
                  # state_E_col, step_E_col):
    """
    for NN:
    "one-hot" --> return the one-hot version of the quaternary tuple
    """
    state_arr = np.concatenate((first_two_actions, state_arr))
    # print("after catting first_two_actions, state_arr = ", state_arr, state_arr.dtype, state_arr.shape)
    state_arr = F.one_hot(torch.from_numpy(state_arr), num_classes=action_depth)
    state_arr = state_arr.numpy()  # q.sample_action expects numpy arr
    # print("one-hot first_two_actions catted state = ")
    # print(state_arr)
    state_arr = np.concatenate((
        # state_E_col,
        # step_E_col,
        state_arr,
        seq_one_hot), axis=1)
    # print("state_arr concat with seq_one_hot, state_E_col, step_E_col =")
    # print(state_arr)
    return state_arr

# environment IDs
# env_id="gym_lattice:Lattice2D-4actionStateEnv-v0"
env_id = "gym_lattice:Lattice2D-3actionStateEnv-v0"

if env_id == "gym_lattice:Lattice2D-4actionStateEnv-v0":
    # observation output mode:
    # 1. "tuple"
    # 2. "index_pental"
    # 3. "index_4N3"
    obs_output_mode = "index_4N3"
elif env_id == "gym_lattice:Lattice2D-3actionStateEnv-v0":
    # observation output mode:
    # 1. "tuple"
    # 2. "index_quaternary"
    # 3. "index_3N2"
    obs_output_mode = "tuple"

# NOTE: partial_reward Sep15 changed to delta of curr-prev rewards

# env = gym.make(id="gym_lattice:Lattice2D-miranda2020Jul-v1", seq=seq)
env = gym.make(
    id=env_id,
    seq=seq,
    obs_output_mode=obs_output_mode,
)

# initial state/observation
# NOTE: env.state != state here
# env.state is actually the chain of OrderedDict([((0, 0), 'H')])
# the state here actually refers to the observation!
initial_state = env.reset()

print("initial state/obs:")
print(initial_state)

# Get number of actions from gym action space
n_actions = env.action_space.n
print("n_actions = ", n_actions)

# choice of network for DRL = "FCN_QNet, RNN_LSTM_onlyLastHidden, BRNN..."
network_choice = "RNN_LSTM_onlyLastHidden"
row_width = action_depth + hp_depth + energy_depth
col_length = env.observation_space.shape[0] + init_HP_len

if network_choice == "FCN_QNet":
    # FCN_QNet() takes two params: insize and outsize
    # insize ==> input size == size of the observation space
    # insize is flattened obs
    insize = col_length * row_width
    print("FCN_QNet insize = ", insize)
    # outsize ==> output size == number of actions
    print("FCN_QNet outsize = ", n_actions)
    q = FCN_QNet(insize, n_actions).to(device)
    q_target = FCN_QNet(insize, n_actions).to(device)
elif network_choice == "RNN_LSTM_onlyLastHidden":
    # config for RNN
    input_size = row_width
    # number of nodes in the hidden layers
    hidden_size = 256
    num_layers = 2
    if "256L2" in q_state_dict_path:
        hidden_size = 256
        num_layers = 2
    elif "512L3" in q_state_dict_path:
        hidden_size = 512
        num_layers = 3

    print("RNN_LSTM_onlyLastHidden with:")
    print(f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}")
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    q = RNN_LSTM_onlyLastHidden(input_size, hidden_size, num_layers, n_actions).to(device)
    q_target = RNN_LSTM_onlyLastHidden(input_size, hidden_size, num_layers, n_actions).to(device)
elif network_choice == "BRNN":
    # config for RNN
    input_size = row_width
    # number of nodes in the hidden layers
    hidden_size = 256
    num_layers = 2

    print("Bidirectional RNN with:")
    print(f"inputs_size={input_size} hidden_size={hidden_size} num_layers={num_layers} num_classes={n_actions}")
    # Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
    q = BRNN(input_size, hidden_size, num_layers, n_actions).to(device)
    q_target = BRNN(input_size, hidden_size, num_layers, n_actions).to(device)

# Saving & Loading Model for Inference
# Save/Load state_dict (Recommended)
# torch.save(q.state_dict(), f'{save_path}{config_str}-state_dict.pth')

q.load_state_dict(torch.load(q_state_dict_path))
print(f"q_state_dict_path: {q_state_dict_path} loaded...")
q.eval()
# call model.eval() to set
# dropout and batch normalization layers
# to evaluation mode before running inference
# we didn't have dropout or BN

# display the model params
count_parameters(q)

# monitor GPU usage
print("torch.cuda.is_available() = ", torch.cuda.is_available())
print("device = ", device)
# Additional Info when using cuda
# https://newbedev.com/how-to-check-if-pytorch-is-using-the-gpu
if device.type == 'cuda':
    # Get the name of the current GPU
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    # print('Memory Usage:')
    # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    # print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# Inspect NN state_dict in pytorch
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in q.state_dict():
    print(param_tensor, "\t", q.state_dict()[param_tensor].size())



for n_episode in range(num_episodes):
    # print("\nEpisode: ", n_episode)

    render = True

    epsilon = 0
    # reset the environment
    # Initialize the environment and state
    s = env.reset()
    print("s = ", s, s.dtype, s.shape)
    print("torch.from_numpy(s) = ", torch.from_numpy(s))

    """
    for NN:
    "one-hot" --> return the one-hot version of the quaternary tuple
    """
    s = one_hot_state(s, seq_one_hot, action_depth)
    print("one-hot initial_state = ")
    print(s)

    done = False
    score = 0.0

    # early stopped due to S_B O and E?
    early_stopped = False
    # whether to avoid F in the next step?
    avoid_F = False

    for step in range(max_steps_per_episode):
        print(f"--- Ep{n_episode} new step-{step}")
        # sample the action from Q
        # convert the given state to torch
        # epsilon is the chance to explore

        # unsqueeze(0) adds a dimension at 0th for batch=1
        # i.e. adds a batch dimension
        a = q.sample_action(torch.from_numpy(s).float().unsqueeze(0), epsilon)
        print('---> action = ', a)

        # take the step and get the returned observation s_prime
        s_prime, r, done, info = env.step(a)
        print(f"s_prime: {s_prime}, reward: {r}, done: {done}, info: {info}")

        # if do not allow for collision, ie. no collision penalty
        # new_state returned from step() will be None
        while s_prime is None:
            # retry until action is not colliding
            print("retry sample another action...")
            a = ((a + 1) % 3)
            print("retried action = ", a)
            # Take the action (a) and observe the outcome state(s') and reward (r)
            s_prime, r, done, info = env.step(a)
            print(f"s_prime: {s_prime}, reward: {r}, done: {done}, info: {info}")

        # Only keep first turn of Left
        # internal 3actionStateEnv self.last_action updated
        a = env.last_action
        print("internal 3actionStateEnv last_action = ", a)

        # Sep19 reward returned from Env is a tuple of (state_E, step_E, reward)
        # print("reward tuple = ", r)
        (state_E, step_E, reward) = r
        print("state_E, step_E, reward = ", state_E, step_E, reward)

        """
        for NN:
            "one-hot" --> return the one-hot version of the quaternary tuple
        """
        # state_E_col[step + init_HP_len] = state_E / OPT_S
        # step_E_col[step + init_HP_len] = step_E / OPT_S
        s_prime = one_hot_state(s_prime, seq_one_hot, action_depth)
                                # state_E_col, step_E_col)
        print("one-hot s_prime = ")
        print(s_prime)

        if info["is_trapped"]:
            print('info["is_trapped"] = ', info["is_trapped"])
            # reward = -(OPT_S - state_E)  # offset by state_E
            # Jan 2022 discover that trap penalty is interfering
            reward = state_E
            # print("adjusted trapped reward = ", reward)

        # NOTE: MUST ENSURE THE REWARD IS FINALIZED BEFORE FEEDING TO RL ALGO!!

        r = reward
        s = s_prime
        score += r

        if render:
            # env.render() to display the graphics
            # set pause_t to see the updates
            env.render(
                display_mode="show",
                pause_t=0,
                save_fig=False,
                score=score,
            )
            print("step-{} render done\n".format(step))
        # clear_output(wait=True)
        # check if the last action ended the episode
        if done:
            # print("Episode finished! Actions: {}".format(info['actions']))
            # if done and used up all actions, pass
            # if done but trapped, zero out the reward
            if len(info['actions']) == (len(seq) - 2):
                # print("Complete: used up all actions!")
                pass
            else:
                # otherwise it means the actions result in trapped episode
                print("TRAPPED!")
            break

    print("score = ", score)
    print(f"\ts_prime: {s_prime[:3], s_prime.shape}, reward: {r}, done: {done}, info: {info}")
