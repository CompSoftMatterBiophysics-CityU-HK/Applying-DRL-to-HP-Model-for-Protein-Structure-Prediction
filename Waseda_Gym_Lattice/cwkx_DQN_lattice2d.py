import argparse
# import csv
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
# from Nov 2021 use PyTorch Official DQN tutorial
# from pytorch_tut_dqn import (
#     device,
#     Transition,
#     ReplayMemory,
#     DQN_FCN,
# )
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

# import the hpsandbox util functions
sys.path.append('../code')
from plot_rewards import (plot_moving_avg,
                          log_rewards_frequency,
                          plot_rewards_histogram,
                          plot_print_rewards_stats,
                          )
from annealing_schedule import (
                            CosineAnnealingSchedule,
                            ExponentialDecay,
                            LinearDecay,
                            Plot_Anneal_Schedule,
)
from early_stop import get_F_patterns, seq_parity_stats, early_stop_S_B


# ***** set up the lattice 2d env *****
# parse CMD arguments
# Parameters starting with - or -- are usually considered optional
parser = argparse.ArgumentParser(
    usage="%(prog)s [seq] [seed] [algo] [num_episodes] [use_early_stop]...",
    description="DQN learning for Lattice 2D HP"
)
parser.add_argument(
    "seq",
)
parser.add_argument(
    "seed",
    type=int,
)
parser.add_argument(
    "algo",
)
parser.add_argument(
    "num_episodes",
    type=int,
)
parser.add_argument(
    "use_early_stop",
    type=int,  # 0 is False, 1 is True
)
args = parser.parse_args()

seq = args.seq.upper()  # Our input sequence
seed = args.seed  # read the seed from CMD
algo = args.algo  # path to save the experiments
num_episodes = args.num_episodes  # number of episodes
use_early_stop = args.use_early_stop  # whether to use early stop

base_dir = f"./{datetime.datetime.now().strftime('%m%d-%H%M')}-"
# construct subdir with seq and seed
config_str = f"{seq[:6]}-{algo}-seed{seed}-{num_episodes}epi"
save_path = base_dir + config_str + "/"

# whether to show or save the matplotlib plots
display_mode = "save"  # save for CMD, show for ipynb
if display_mode == "save":
    save_fig = True
else:
    save_fig = False

# for local optima inspection
local_optima = {21, 23}  # set of local optimas to inspect
optima_idx = {}  # for env.render() r_max, count local optima index
optima_actions_set = {}
for lo in local_optima:
    optima_idx[lo] = 0
    optima_actions_set[lo] = []

# create the folder according to the save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
for lo in local_optima:
    if not os.path.exists(os.path.join(save_path, f"Score_{lo}")):
        os.makedirs(os.path.join(save_path, f"Score_{lo}"))

# Redirect 'print' output to a file in python
orig_stdout = sys.stdout
f = open(save_path + 'out.txt', 'w')
sys.stdout = f

# apply flush=True to every print function call in the module with a partial function
from functools import partial
print = partial(print, flush=True)

# log the system hostname
print("os.uname() = ", os.uname())

print("args parse seq = ", seq)
print("args parse seed = ", seed)
print("args parse algo = ", algo)
print("args parse save_path = ", save_path)
print("args parse num_episodes = ", num_episodes)
print("args parse use_early_stop = ", use_early_stop)

print("optima_idx = ", optima_idx)
print("optima_actions_set = ", optima_actions_set)

# @hyperparameters
max_steps_per_episode = len(seq)

learning_rate = 0.0005

mem_start_train = max_steps_per_episode * 50  # for memory.size() start training
TARGET_UPDATE = 100  # fix to 100

# capped at 50,000 for <=48mer
buffer_limit = int(min(50000, num_episodes // 10))  # replay-buffer size

print("##### Summary of Hyperparameters #####")
print("learning_rate: ", learning_rate)
print("BATCH_SIZE: ", batch_size)
print("GAMMA: ", gamma)
print("mem_start_train: ", mem_start_train)
print("TARGET_UPDATE: ", TARGET_UPDATE)
print("buffer_limit: ", buffer_limit)
print("train_times: ", train_times)
print("##### End of Summary of Hyperparameters #####")

# Exploration parameters
max_exploration_rate = 1
min_exploration_rate = 0.01

# render settings
show_every = num_episodes // 1000  # for plot_print_rewards_stats
pause_t = 0.0
# metric for evaluation
rewards_all_episodes = np.zeros(
    (num_episodes,),
    # dtype=np.int32
)
reward_max = 0
# keep track of trapped SAW
num_trapped = 0
# early_stopped
num_early_stopped = 0

warmRestart = True
decay_mode = "exponential"  # exponential, cosine, linear
num_restarts = 1  # for cosine decay warmRestart=True
exploration_decay_rate = 5  # for exponential decay
start_decay = 0  # for exponential and linear
print(f"decay_mode={decay_mode} warmRestart={warmRestart}")
print(f"num_restarts={num_restarts} exploration_decay_rate={exploration_decay_rate} start_decay={start_decay}")
# visualize the annealing schedule
Plot_Anneal_Schedule(
    num_episodes,
    min_exploration_rate,
    max_exploration_rate,
    mode=display_mode,
    save_path=save_path,
    warmRestart=warmRestart,
    decay_mode=decay_mode,
    num_restarts=num_restarts,
    exploration_decay_rate=exploration_decay_rate,
    start_decay=start_decay,
)

# for early stop schemes
# scheme 1: based on F patterns
(
    N_half,
    F_half_pattern,
    F_half_minus_one_pattern,
) = get_F_patterns(seq)

print(f"N_half={N_half}\nF_half_pattern={F_half_pattern}\nF_half_minus_one_pattern={F_half_minus_one_pattern}")

condensed_F_pattern = F_half_minus_one_pattern.replace(", ", '')
print("condensed_F_pattern = ", condensed_F_pattern)

# scheme 2: based on O_S_B
(
    Odd_H_indices,
    Even_H_indices,
    O_S,
    E_S,
    O_terminal_H,
    E_terminal_H,
    OPT_S,
) = seq_parity_stats(seq)



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

# reproducible environment and action spaces, do not change lines 6-11 here (tools > settings > editor > show line numbers)
torch.manual_seed(seed)
env.seed(seed)
random.seed(seed)
np.random.seed(seed)
env.action_space.seed(seed)

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

q_target.load_state_dict(q.state_dict())

# display the model params
count_parameters(q)

optimizer = optim.Adam(q.parameters(), lr=learning_rate)

memory = ReplayBuffer(buffer_limit)
# load pre-populated Replay Buffer with good early-stopped and r_max
# TODO: memory.load(f'./xxx.pkl')

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

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# time the experiment
start_time = time()
time_log("Start RL Program")

for n_episode in range(num_episodes):
    # print("\nEpisode: ", n_episode)

    # only render the game every once a while
    if (n_episode == 0) or ((n_episode+1) % show_every == 0):
        if display_mode == "show":
            render = True  # can enable render for debugging
        elif display_mode == "save":
            render = False
    else:
        render = False

    # epsilon = max(min_exploration_rate, max_exploration_rate - exploration_decay_rate*(n_episode/200)) # linear annealing
    if decay_mode == "cosine":
        epsilon = CosineAnnealingSchedule(
            n_episode,
            num_episodes,
            min_exploration_rate,
            max_exploration_rate,
            warmRestart=warmRestart,
            num_restarts=num_restarts,
        )
    elif decay_mode == "exponential":
        epsilon = ExponentialDecay(
            n_episode,
            num_episodes,
            min_exploration_rate,
            max_exploration_rate,
            exploration_decay_rate=exploration_decay_rate,
            start_decay=start_decay,
        )
    elif decay_mode == "linear":
        epsilon = LinearDecay(
            n_episode,
            num_episodes,
            min_exploration_rate,
            max_exploration_rate,
            start_decay=start_decay,
        )

    # reset the environment
    # Initialize the environment and state
    s = env.reset()
    # print("s = ", s, s.dtype, s.shape)
    # print("torch.from_numpy(s) = ", torch.from_numpy(s))

    """
    for NN:
    "one-hot" --> return the one-hot version of the quaternary tuple
    """
    # revert state_E and step_E cols Dec03 2021
    # # column of state_E and column of step_e
    # state_E_col = np.zeros((col_length, 1))
    # step_E_col = np.zeros((col_length, 1))
    s = one_hot_state(s, seq_one_hot, action_depth)
                      # state_E_col, step_E_col)
    # print("one-hot initial_state = ")
    # print(s)

    done = False
    score = 0.0

    # early stopped due to S_B O and E?
    early_stopped = False
    # whether to avoid F in the next step?
    avoid_F = False

    for step in range(max_steps_per_episode):
        # print(f"--- Ep{n_episode} new step-{step}")
        # sample the action from Q
        # convert the given state to torch
        # epsilon is the chance to explore

        # unsqueeze(0) adds a dimension at 0th for batch=1
        # i.e. adds a batch dimension
        a = q.sample_action(torch.from_numpy(s).float().unsqueeze(0), epsilon)
        # print('---> action = ', a)

        if use_early_stop:
            # print("avoid_F = ", avoid_F)
            if a == 1 and avoid_F:
                # print("do not use F in this step!")
                a = np.random.choice([0, 2])
                # print("new sampled action is = ", a)
            # reset avoid_F
            avoid_F = False

        # take the step and get the returned observation s_prime
        s_prime, r, done, info = env.step(a)
        # print(f"s_prime: {s_prime}, reward: {r}, done: {done}, info: {info}")

        # if do not allow for collision, ie. no collision penalty
        # new_state returned from step() will be None
        while s_prime is None:
            # retry until action is not colliding
            # print("retry sample another action...")
            a = ((a + 1) % 3)
            # print("retried action = ", a)
            # Take the action (a) and observe the outcome state(s') and reward (r)
            s_prime, r, done, info = env.step(a)
            # print(f"s_prime: {s_prime}, reward: {r}, done: {done}, info: {info}")

        # Only keep first turn of Left
        # internal 3actionStateEnv self.last_action updated
        a = env.last_action
        # print("internal 3actionStateEnv last_action = ", a)

        # Sep19 reward returned from Env is a tuple of (state_E, step_E, reward)
        # print("reward tuple = ", r)
        (state_E, step_E, reward) = r
        # print("state_E, step_E, reward = ", state_E, step_E, reward)

        """
        for NN:
            "one-hot" --> return the one-hot version of the quaternary tuple
        """
        # state_E_col[step + init_HP_len] = state_E / OPT_S
        # step_E_col[step + init_HP_len] = step_E / OPT_S
        s_prime = one_hot_state(s_prime, seq_one_hot, action_depth)
                                # state_E_col, step_E_col)
        # print("one-hot s_prime = ")
        # print(s_prime)

        if info["is_trapped"]:
            # print('info["is_trapped"] = ', info["is_trapped"])
            # reward = -(OPT_S - state_E)  # offset by state_E
            # Jan 2022 discover that trap penalty is interfering
            reward = state_E
            # print("adjusted trapped reward = ", reward)

        if use_early_stop and not done:
            # Early Stop Scheme
            # Scheme 1: F-patterns based on floor(N/2)-1's F
            # NOTE: dont use penalty, use manual guiding
            # print('info["actions"] = ', info["actions"])
            info_actions_str = ''.join(info["actions"])
            # print("info_actions_str = ", info_actions_str)
            info_actions_str_with_F = info_actions_str + 'F'
            # print("info_actions_str_with_F = ", info_actions_str_with_F)
            # print('condensed_F_pattern in info_actions_str_with_F? = ', condensed_F_pattern in info_actions_str_with_F)
            if condensed_F_pattern in info_actions_str_with_F:
                # print(f"condensed_F_pattern({condensed_F_pattern}) in info_actions_str_with_F...")
                # print("do not go F in the next step")
                avoid_F = True
            # Scheme 2: based on S_B remaining rewards
            max_delta = early_stop_S_B(seq, step+1, O_S, E_S,
                            Odd_H_indices, Even_H_indices,
                            O_terminal_H, E_terminal_H, OPT_S)
            # important to keep hitting the episodes >= reward_max for q-values
            if (state_E + max_delta) < reward_max:
                # print(f"(state_E + max_delta)={(state_E + max_delta)} < reward_max({reward_max})")
                reward = state_E  # early_stop_penalty
                # print("adjusted ES reward = ", reward)
                done = True
                early_stopped = True

        # NOTE: MUST ENSURE THE REWARD IS FINALIZED BEFORE FEEDING TO RL ALGO!!

        r = reward

        # NOTE: done_mask is for when you get the end of a run,
        # then is no future reward, so we mask it with done_mask
        done_mask = 0.0 if done else 1.0
        # put the sampled state-action-reward-stateAfterwards
        # NOTE: here r is divided by 100.0? Undo the 100.0 Sep04 2021
        # NOTE: priorize experience that are
        # 1. not trapped (r = -1)
        # 2. not early-stopped
        # 3.
        memory.put((s,a,r,s_prime, done_mask))
        s = s_prime

        # Add new reward
        # NOTE: Sep15 update partial_reward to be delta instead of progress*curr_reward
        # NOTE: Sep19 update reward to be a tuple, and reward is 0 until done
        score += r

        if render:
            # env.render() to display the graphics
            # set pause_t to see the updates
            env.render(
                display_mode=display_mode,
                pause_t=pause_t,
                save_fig=save_fig,
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
                if use_early_stop and early_stopped:
                    # print("EARLY STOPPED")
                    num_early_stopped += 1
                else:
                    # otherwise it means the actions result in trapped episode
                    # print("TRAPPED!")
                    num_trapped += 1
            break

    # eventually if memory is big enough, we start running the training loop
    # start training after 2000 (for eg) can get a wider distribution
    # print("memory.size() = ", memory.size())
    if memory.size()>mem_start_train:
        train(q, q_target, memory, optimizer)

    # Update the target network, copying all weights and biases in DQN
    if n_episode % TARGET_UPDATE == 0:
        q_target.load_state_dict(q.state_dict())

    # print("score = ", score)
    # inspect local optima for non-ES spisode scores
    if (score in local_optima) and (len(info['actions']) == (len(seq) - 2)):
        optima_info_actions_str = ''.join(info["actions"])
        # print("optima_info_actions_str = ", optima_info_actions_str)
        if optima_info_actions_str not in optima_actions_set[score]:
            # print(">>> is a new local optima conf<<<")
            env.render(
                display_mode=display_mode,
                save_fig=save_fig,
                save_path=f"{save_path}Score_{int(score)}",
                score=score,
                optima_idx=optima_idx[score],
            )
            optima_idx[score] += 1
            optima_actions_set[score].append(optima_info_actions_str)
            # print("optima_idx = ", optima_idx)
            # print("optima_actions_set = ", optima_actions_set)
    # Add current episode reward to total rewards list
    rewards_all_episodes[n_episode] = score
    # update max reward found so far
    if score > reward_max:
        print("found new highest reward = ", score)
        reward_max = score
        env.render(
            display_mode=display_mode,
            save_fig=save_fig,
            save_path=save_path,
            score=score,
        )

    if (n_episode == 0) or ((n_episode+1) % show_every == 0):
        print("Episode {}, score: {:.1f}, epsilon: {:.2f}, reward_max: {}".format(
            n_episode,
            score,
            epsilon,
            reward_max,
        ))
        print(f"\ts_prime: {s_prime[:3], s_prime.shape}, reward: {r}, done: {done}, info: {info}")
    # move on to the next episode

print('Complete')
# for time records
end_time = time()
elapsed = end_time - start_time
time_log("End Program", secondsToStr(elapsed))

# Save the rewards_all_episodes with numpy save
with open(f'{save_path}{config_str}-rewards_all_episodes.npy', 'wb') as f:
    np.save(f, rewards_all_episodes)

# Save the ReplayMemory
# TODO: later memory.save(f'{save_path}{config_str}-replaybuffer.pkl')

# Save the pytorch model
# Saving & Loading Model for Inference
# Save/Load state_dict (Recommended)
torch.save(q.state_dict(), f'{save_path}{config_str}-state_dict.pth')

# ***** plot the stats and save in save_path *****

plot_moving_avg(rewards_all_episodes, mode=display_mode, save_path=save_path)
log_rewards_frequency(rewards_all_episodes)
plot_rewards_histogram(
    rewards_all_episodes,
    mode=display_mode,
    save_path=save_path,
    config_str=config_str,
)
plot_print_rewards_stats(
    rewards_all_episodes,
    show_every,
    args,
    mode=display_mode,
    save_path=save_path,
)

env.close()

print("optima_idx = ", optima_idx)
print("optima_actions_set = ", optima_actions_set)

print("\nnum_trapped = ", num_trapped)
if use_early_stop:
    print("num_early_stopped = ", num_early_stopped)
# last line of the output is the max reward
print("\nreward_max = ", reward_max)

# Redirect 'print' output to a file in python
sys.stdout = orig_stdout
f.close()

# # record the performance to a mega stats csv
# with open(f"./HP{len(seq)}-{base_dir}-stats.csv", "a+") as csv_file:
#     stats_writer = csv.writer(
#         csv_file,
#         delimiter=',',
#         quotechar='"',
#         quoting=csv.QUOTE_MINIMAL
#     )
#     stats_writer.writerow([seq, seed, reward_max])
