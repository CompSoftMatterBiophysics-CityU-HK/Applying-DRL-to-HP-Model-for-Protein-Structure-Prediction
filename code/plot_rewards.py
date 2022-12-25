import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator)


def plot_moving_avg(scores, n=500, mode="show", save_path=""):
    print("means = ", scores.mean())

    # useful utility function for graphing the average
    def moving_average(a, n=n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.plot(moving_average(scores, n=n))
    # plt.plot(moving_average(opt_scores, n=500))
    # plt.plot(moving_average(rand_scores, n=500))
    if mode == "show":
        plt.show()
    elif mode == "save":
        # save the pdf fig with seq name
        plt.savefig(save_path + "moving_avg-" + str(n) + ".png")
    plt.close()


def log_rewards_frequency(rewards_all_episodes):
    # review all episodes' rewards
    print("$$$ rewards_all_episodes: ", rewards_all_episodes)
    print("$$$ rewards_all_episodes last 10 rewards = ",
          rewards_all_episodes[-10:])
    # count the frequency of unique rewards
    unique_elements, counts_elements = np.unique(
                                            rewards_all_episodes,
                                            return_counts=True)
    print("Frequency of unique rewards of rewards_all_episodes:")
    with np.printoptions(suppress=True):
        print(np.asarray((unique_elements, counts_elements)))


def plot_rewards_histogram(rewards_all_episodes, mode="show", save_path="", config_str=""):
    # plot the histogram of rewards_all_episodes
    # number of bins derived from https://stackoverflow.com/questions/30112420/histogram-for-discrete-values-with-matplotlib
    data = rewards_all_episodes
    d = np.diff(np.unique(data)).min()
    left_of_first_bin = data.min() - float(d)/2
    right_of_last_bin = data.max() + float(d)/2

    # Width, height in inches.
    # default: [6.4, 4.8]
    fig_width = 6.4
    fig_height = 4.8
    # adjust the height of the histogram
    if right_of_last_bin - left_of_first_bin > 10:
        fig_width = 8
        fig_height = 5
    elif right_of_last_bin - left_of_first_bin > 20:
        fig_width = 12
        fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # the histogram of the data
    n, bins, patches = ax.hist(
        data,
        np.arange(left_of_first_bin, right_of_last_bin + d, d),
        density=True,
        facecolor='g',
    )

    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '.0f' formatting but don't label
    # minor ticks.  The string is used directly, the `StrMethodFormatter` is
    # created automatically.
    if right_of_last_bin - left_of_first_bin > 20:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(1))  # multiple of 1
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_xlabel('Rewards')
    ax.set_ylabel('Frequency (Num of Episodes)')
    ax.set_title(f'Histogram of Rewards: {config_str}')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.grid(True)
    if mode == "show":
        plt.show()
    elif mode == "save":
        # save the pdf fig with seq name
        plt.savefig(save_path+"rewards_histogram.png")
    plt.close()


def plot_print_rewards_stats(rewards_all_episodes,
                             show_every,
                             args,
                             mode="show",
                             save_path=""):
    # unpack the args
    seq = args.seq
    seed = args.seed
    algo = args.algo
    num_episodes = args.num_episodes

    # Calculate and print the average reward per show_every episodes
    rewards_per_N_episodes = np.split(
                                np.array(rewards_all_episodes),
                                num_episodes/show_every
                            )
    count = show_every

    # for plotting
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

    print("\n********Stats per {} episodes********\n".format(show_every))
    for r in rewards_per_N_episodes:
        # print(count, "avg: ", str(sum(r/show_every)))
        # print(count, "min: ", str(min(r)))
        # print(count, "max: ", str(max(r)))

        aggr_ep_rewards['ep'].append(count)
        aggr_ep_rewards['avg'].append(sum(r/show_every))
        aggr_ep_rewards['min'].append(min(r))
        aggr_ep_rewards['max'].append(max(r))

        count += show_every

    # Width, height in inches.
    # default: [6.4, 4.8]
    fig_width = 6.4
    fig_height = 4.8
    # adjust the height of the histogram
    if np.array(rewards_all_episodes).max() - np.array(rewards_all_episodes).min() > 10:
        fig_width = 6.5
        fig_height = 6.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # Be sure to only pick integer tick locations
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.set_xlabel('Episode Index')
    ax.set_ylabel('Episode Reward')

    ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
    ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
    ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)

    # split the seq into chunks of 10 for the matplotlib title
    chunks, chunk_size = len(seq), 10
    seq_title_list = [
        seq[i:i+chunk_size]+"\n" for i in range(0, chunks, chunk_size)
    ]
    seq_title_str = ''.join(seq_title_list)
    title = "{}Algo={}, Epi={}, Seed={}\nShow-every {}".format(
        seq_title_str,
        algo,
        num_episodes,
        seed,
        show_every,
    )
    # print("Title: ", title)
    plt.title(title)
    plt.grid(True, which="major", lw=1.2, linestyle='-')
    plt.grid(True, which="minor", lw=0.8, linestyle='--')
    plt.tight_layout()
    if mode == "show":
        plt.show()
    elif mode == "save":
        # save the pdf fig with seq name
        plt.savefig("{}Seq_{}-{}-Eps{}-Seed{}.png".format(
            save_path,  # "./xxx"
            seq,
            algo,
            num_episodes,
            seed,
        ))
    plt.close()


def extract_max_per_chunk(rewards_all_episodes,
                          num_episodes,
                          show_every):
    """
    extract the max per chunk
    """
    rewards_per_N_episodes = np.split(
                                rewards_all_episodes,
                                num_episodes//show_every
                            )
    aggr_max = np.zeros((num_episodes//show_every,))

    for index, r in enumerate(rewards_per_N_episodes):
        aggr_max[index] = np.amax(r)

    return aggr_max

def avg_std_of_max(seed_42_data,
                   seed_1984_data,
                   seed_1991_data,
                   seed_2021_data,
                   num_episodes,
                   N_chunks,
                   show_every,
                   ):
    max_seed_42_data = extract_max_per_chunk(seed_42_data,
                                            num_episodes,
                                            show_every)
    max_seed_1984_data = extract_max_per_chunk(seed_1984_data,
                                            num_episodes,
                                            show_every)
    max_seed_1991_data = extract_max_per_chunk(seed_1991_data,
                                            num_episodes,
                                            show_every)
    max_seed_2021_data = extract_max_per_chunk(seed_2021_data,
                                            num_episodes,
                                            show_every)

    print("max_data shape = ", max_seed_42_data.shape, max_seed_2021_data.shape)

    num_seeds = 4
    max_stacked = np.zeros((num_seeds, N_chunks))

    max_stacked[0] = max_seed_42_data
    max_stacked[1] = max_seed_1984_data
    max_stacked[2] = max_seed_1991_data
    max_stacked[3] = max_seed_2021_data

    avg = np.mean(max_stacked, axis=0)
    std = np.std(max_stacked, axis=0)

    # print("avg = ", avg)
    # print("std = ", std)

    return avg, std

def plot_shaded_std(title, N_chunks, show_every, mean_1, std_1, mean_2, std_2,
                    mean_3=None, std_3=None):
    # import seaborn as sns
    # sns.set()

    import matplotlib.pyplot as plt

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=24)     # fontsize of the axes title
    plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=21)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=21)    # fontsize of the tick labels
    plt.rc('legend', fontsize=15)    # legend fontsize
    # plt.rc('figure', titlesize=48)  # fontsize of the figure title
    plt.rcParams["figure.figsize"] = (8, 7)
    # plt.rcParams["font.family"] = "monospace"

    x = np.arange(N_chunks)
    x = x*show_every
    x = x // 1000  # per K
    # print(x)

    # plot
    fig, ax = plt.subplots()

    ax.plot(x, mean_1, color='b', label='RAND')
    ax.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
    ax.plot(x, mean_2, color='r', label='DQN-LSTM')
    # ax.plot(x, mean_2, color='r', label='DQN+Pruning+TrapPenalty')
    ax.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
    ax.plot(x, mean_3, color='y', label='DQN-FCN')
    # ax.plot(x, mean_3, color='y', label='DQN w/o Pruning/TrapPenalty')
    ax.fill_between(x, mean_3 - std_3, mean_3 + std_3, color='y', alpha=0.2)

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}k"))

    # optional: set the y limit
    # ax.set_ylim(-20, -9)

    ax.set_xlabel("Episode Index")
    ax.set_ylabel("Energy")
    ax.legend(loc='lower left')
    # ax.legend(loc='upper right')
    ax.set_title(title, fontsize=27)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.savefig(title + "-RANDvsDQN.png")
    plt.show()
    plt.close()
