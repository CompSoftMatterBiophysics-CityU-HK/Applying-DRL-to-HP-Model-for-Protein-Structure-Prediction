import gym
import numpy as np
import os


# sequence
seq = "HHPPHPPHPPHPPHPPHPPHPPHH"

optimal_list = []
local_optima = 9
save_path = f"./{len(seq)}mer_E{local_optima}_set"

N = 24

assert N == len(seq)

# open file and read the content in a list
with open(f'./confs_{len(seq)}mer_E{local_optima}.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]

        # add current item to the list
        optimal_list.append(x)

# display list
print(optimal_list)


ACTION_STR_TO_NUM = {
    'L': 0,
    'F': 1,
    'R': 2,
}

# create the folder according to the save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)


# environment IDs
env_id = "gym_lattice:Lattice2D-3actionStateEnv-v0"
obs_output_mode = "tuple"
env = gym.make(
    id=env_id,
    seq=seq,
    obs_output_mode=obs_output_mode,
)

display_mode = "save"
save_fig = True
render = True

for ol_index, actions in enumerate(optimal_list):
    print(f"\n{ol_index}-th actions item = ", actions)
    # reset the environment
    # Initialize the environment and state
    env.reset()
    done = False
    score = 0.0

    for i, action in enumerate(actions):
        # print(f"\n{i}-th action = ", action)
        a = ACTION_STR_TO_NUM[action]
        # print("ACTION_STR_TO_NUM -> ", a)
        s_prime, r, done, info = env.step(a)
        # print(f"s_prime: {s_prime}, reward: {r}, done: {done}, info: {info}")

        # print("reward tuple = ", r)
        (state_E, step_E, reward) = r
        # print("state_E, step_E, reward = ", state_E, step_E, reward)
        r = reward
        score += r

        if done:
            break
    print("score = ", score)
    assert int(score) == local_optima
    env.render(
        display_mode=display_mode,
        save_fig=save_fig,
        save_path=f"{save_path}",
        score=local_optima,
        optima_idx=ol_index,
    )

print('Complete')
