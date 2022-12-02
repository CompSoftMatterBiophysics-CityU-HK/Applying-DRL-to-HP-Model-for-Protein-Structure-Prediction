import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.ticker as plticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator)


def CosineAnnealing(epoch, T_input, eta_min=0.01, eta_max=1.0, warmRestart=True):
    """
    T_input for T_{0,i,cur,max} etc
    warmRestart: whether to have automatic cyclic warmRestart
    """
    if warmRestart:
        # T_0: # iterations for the 1st restart
        T_0 = T_input
        # T_i: # epochs between restarts
        T_i = T_0
        # T_cur: # epochs since last restart
        if epoch >= T_0:
            T_cur = epoch % T_0
        else:
            T_i = T_0
            T_cur = epoch
        eta_t = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
    else:
        # T_max: # epochs between restarts
        T_max = T_input
        # T_cur: # epochs since last restart
        T_cur = epoch

        eta_t = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * T_cur / T_max)) / 2
    return eta_t

def CosineAnnealingSchedule(epoch, num_episodes, eta_min=0.01, eta_max=1.0, warmRestart=True, num_restarts=10):
    if warmRestart:
        # number of restarts
        eta_t = CosineAnnealing(epoch, num_episodes//num_restarts, eta_min, eta_max, warmRestart)
    else:
        T_0 = math.floor(num_episodes * 0.3)
        T_1 = math.floor(num_episodes * 0.1)
        T_2 = math.floor(num_episodes * 0.2)
        T_3 = math.floor(num_episodes * 0.4)
        if epoch <= T_0:
            eta_t = CosineAnnealing(epoch, T_0, eta_min, eta_max, warmRestart)
        elif epoch > T_0 and epoch <= (T_0+T_1):
            eta_t = CosineAnnealing(epoch-(T_0), T_1, eta_min, eta_max, warmRestart)
        elif epoch > T_1 and epoch <= (T_0+T_1+T_2):
            eta_t = CosineAnnealing(epoch-(T_0+T_1), T_2, eta_min, eta_max, warmRestart)
        elif epoch > T_2:
            eta_t = CosineAnnealing(epoch-(T_0+T_1+T_2), T_3, eta_min, eta_max, warmRestart)
    
    return eta_t

def ExponentialDecay(episode, num_episodes,
                min_exploration_rate, max_exploration_rate,
                exploration_decay_rate=5,
                start_decay=0):
    decay_duration = num_episodes - start_decay
    exploration_rate = max_exploration_rate
    if episode > start_decay:
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*(episode-start_decay)/decay_duration)
    return exploration_rate

def LinearDecay(episode, num_episodes,
                min_exploration_rate, max_exploration_rate,
                start_decay=0):
    decay_duration = num_episodes - start_decay
    exploration_rate = max_exploration_rate
    if episode > start_decay:
        exploration_rate = min_exploration_rate + \
                            (decay_duration-(episode-start_decay))/decay_duration
        # print("(decay_duration-(episode-start_decay))/decay_duration = ", (decay_duration-(episode-start_decay))/decay_duration)
    # print("exploration_rate = ", exploration_rate)
    return exploration_rate


def Plot_Anneal_Schedule(num_episodes, eta_min=0.01, eta_max=1.0, mode="save", save_path="", warmRestart=True,
                         decay_mode="cosine",
                         num_restarts=10,
                         exploration_decay_rate=5,
                         start_decay=0):
    print("warmRestart = ", warmRestart)
    # Exploration parameters
    max_exploration_rate = eta_max
    min_exploration_rate = eta_min

    y =np.zeros(
        (num_episodes,),
    )

    if decay_mode == "cosine":
        for n_episode in range(num_episodes):
            y[n_episode] = CosineAnnealingSchedule(
                n_episode,
                num_episodes,
                min_exploration_rate,
                max_exploration_rate,
                warmRestart=warmRestart,
                num_restarts=num_restarts,
            )
    elif decay_mode == "exponential":
        for n_episode in range(num_episodes):
            y[n_episode] = ExponentialDecay(
                n_episode,
                num_episodes,
                min_exploration_rate,
                max_exploration_rate,
                exploration_decay_rate=exploration_decay_rate,
                start_decay=start_decay,
            )
    elif decay_mode == "linear":
        for n_episode in range(num_episodes):
            y[n_episode] = LinearDecay(
                n_episode,
                num_episodes,
                min_exploration_rate,
                max_exploration_rate,
                start_decay=start_decay,
            )

    fig, ax = plt.subplots()

    plt.yticks(np.arange(-.1, 1.1, 0.1))

    ax.plot(np.arange(num_episodes), y)
    ax.set(xlabel='Episode', ylabel='epsilon',
        # title=f'Epsilon Decay with {decay_mode}Decay warmRestart={warmRestart}')
        title=f'Epsilon Decay with {decay_mode}Decay')
    ax.grid()

    if mode == "show":
        plt.show()
    elif mode == "save":
        # save the pdf fig with seq name
        plt.savefig(save_path + "Anneal_Schedule_" + str(num_episodes) + "_episodes.png")
    plt.close()
