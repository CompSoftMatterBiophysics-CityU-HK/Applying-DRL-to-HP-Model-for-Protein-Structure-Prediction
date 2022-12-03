import argparse
import sys
import os
# import csv

import numpy as np

# use gym.make() to init envs
import gym

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

from early_stop import get_F_patterns, seq_parity_stats, early_stop_S_B

import datetime


# parse CMD arguments
# Parameters starting with - or -- are usually considered optional
parser = argparse.ArgumentParser(
    usage="%(prog)s [seq] [seed] [algo] [num_episodes] [use_early_stop]...",
    description="Random Actions for Lattice 2D HP"
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

# create the folder according to the save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

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
    obs_output_mode = "tuple"  # use tuple for large N, otherwise index_3N2 memory overflow

# NOTE: partial_reward Sep15 changed to delta of curr-prev rewards

# env = gym.make(id="gym_lattice:Lattice2D-miranda2020Jul-v1", seq=seq)
env = gym.make(
    id=env_id,
    seq=seq,
    obs_output_mode=obs_output_mode,
)

# env.seed() will seed the environment randomness
env.seed(seed)

# initial state/observation
# NOTE: env.state != state here
# env.state is actually the chain of OrderedDict([((0, 0), 'H')])
# the state here actually refers to the observation!
initial_state = env.reset()

print("initial state/obs:")
print(initial_state)

print("\n********************* main program **********************\n")

# @hyperparameters
max_steps_per_episode = len(seq)

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

# penalty scheme for early stop and trapped
# constant_early_stop_penalty = False
constant_trap_penalty = True

# constant base for trap penalty vs progressive penalty
if constant_trap_penalty:
    # trap_penalty_epi = -1 * len(seq)
    # trap_penalty_epi = 0
    trap_penalty_epi = -1 * OPT_S  # change to -1*OPT(S) Dec2021
else:
    trap_penalty_per_node = -2  # punish trapped based on folded length

# time the experiment
start_time = time()
time_log("Start random Program")

# all random algorithm
for episode in range(num_episodes):
    # print("\nNew Episode ", episode)

    # only render the game every once a while
    if (episode == 0) or ((episode+1) % show_every == 0):
        if display_mode == "show":
            render = True  # can enable render for debugging
        elif display_mode == "save":
            render = False
    else:
        render = False

    # initialize new episode params
    state = env.reset()

    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # print(f"--- Ep{episode} new step-{step}")

        # Exploration-exploitation trade-off
        action = env.action_space.sample()
        # print("---> action = ", action)
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)
        # print(f"new_state: {new_state}, reward: {reward}, done: {done}, info: {info}")

        # if do not allow for collision, ie. no collision penalty
        # new_state returned from step() will be None
        while new_state is None:
            # retry until action is not colliding
            # print("retry sample another action...")
            action = ((action + 1) % 3)
            # action = env.action_space.sample()
            # print("retried action = ", action)
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)
            # print(f"new_state: {new_state}, reward: {reward}, done: {done}, info: {info}")
        # Get correspond q value from state, action pair

        # Only keep first turn of Left
        # internal 3actionStateEnv self.last_action updated
        action = env.last_action
        # print("internal 3actionStateEnv last_action = ", action)

        # Sep19 reward returned from Env is a tuple of (state_E, step_E, reward)
        # print("reward tuple = ", reward)
        (state_E, step_E, reward) = reward
        # print("state_E, step_E, reward = ", state_E, step_E, reward)

        if info["is_trapped"]:
            # print('info["is_trapped"] = ', info["is_trapped"])
            if constant_trap_penalty:
                # instead of trap_penalty*len, use a clear-out penalty
                # reward = trap_penalty_epi
                reward = trap_penalty_epi + state_E  # offset by state_E
            else:
                # NOTE: from Sep 23, use a trap penalty per node * folded lenth + state_E
                reward = trap_penalty_per_node*info["chain_length"] + state_E
            # print("adjusted trapped reward = ", reward)

        # Set new state
        state = new_state
        # Add new reward
        # NOTE: Sep15 update partial_reward to be delta instead of progress*curr_reward
        # NOTE: Sep19 update reward to be a tuple, and reward is 0 until done
        rewards_current_episode += reward

        if render:
            # env.render() to display the graphics
            # set pause_t to see the updates
            env.render(
                display_mode=display_mode,
                pause_t=pause_t,
                save_fig=save_fig,
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
                # print("TRAPPED!")
                # # instead of trap_penalty*len, use a clear-out penalty
                # rewards_current_episode = -1  # mark trapped as -1 for debugging
                num_trapped += 1
            break

    # print("rewards_current_episode = ", rewards_current_episode)
    # Add current episode reward to total rewards list
    rewards_all_episodes[episode] = rewards_current_episode
    # update max reward found so far
    if rewards_current_episode > reward_max:
        print("found new highest reward = ", rewards_current_episode)
        reward_max = rewards_current_episode
        env.render(
            display_mode=display_mode,
            save_fig=save_fig,
            save_path=save_path,
            score=rewards_current_episode,
        )

    if (episode == 0) or ((episode+1) % show_every == 0):
        print("Episode {}, episode reward {}, epsilon: {:.2f}, reward_max: {}".format(
            episode,
            rewards_current_episode,
            1,
            reward_max,
        ))
        print(f"\tnew_state: {new_state}, reward: {reward}, done: {done}, info: {info}")
    # move on to the next episode

print('Complete')
# for time records
end_time = time()
elapsed = end_time - start_time
time_log("End Program", secondsToStr(elapsed))

# Save the rewards_all_episodes with numpy save
with open(f'{save_path}{config_str}-rewards_all_episodes.npy', 'wb') as f:
    np.save(f, rewards_all_episodes)

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

print("\nnum_trapped = ", num_trapped)
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
