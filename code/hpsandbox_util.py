import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from sklearn.metrics.pairwise import euclidean_distances
import io


def label_conf_seq(conf_seq, conf_title):
    """
    input:
        conf_seq:
            the conf sequence as taken from HPSandbox file
            list of coordinates --> [(x1,y1), (x3,y2)...]
        conf_title: the conf file title
    output:
        conf with the H/P sequence
        list of tuples
        [((0, 0), 'H'),
        ((0, 1), 'H'),...
        ((3, 1), 'P')]
    NOTE: the output from label_conf_seq() is the same
    as the OrderedDict's items()
    """
    str_seq = conf_title[:-5]

    # the sequence length must match
    assert len(str_seq) == len(conf_seq)

    labelled_conf = []
    for index, letter in enumerate(str_seq):
        labelled_conf.append((conf_seq[index], letter))

    return labelled_conf


# map of LFR directions and ints
LFR_map = {
    "left": 0,
    "forward": 1,
    "right": 2,
    "error": -1,
    "0": "left",
    "1": "forward",
    "2": "right",
    "-1": "error",
}


def move_LFR_direction(p1, p2, move_direction):
    """
    move in Left, Forward, Right directions
    input:
        p1 is the point two positions earlier
        p2 is the point one position earlier
        both p1 and p2 are tuples of (x,y)

        move_direction:
        left: 0,
        forward: 1,
        right: 2,
        # error: -1

    return:
        p3 is to-be-moved point
    """
    if move_direction not in {0, 1, 2}:
        print("ILLEGAL MOVE")
        return

    # print("p1, p2: ", p1, p2)
    x1, y1 = p1
    x2, y2 = p2

    # candidates adjacent_coords
    p3_candidates = [
        (x2 - 1, y2),
        (x2, y2 - 1),
        (x2, y2 + 1),
        (x2 + 1, y2),
    ]
    # print("p3_candidates = ", p3_candidates)

    for candidate in p3_candidates:
        # print("try p3 candidate ", candidate)
        direction = derive_LFR_direction(p1, p2, candidate)
        # print("direction = ", direction)
        if direction == move_direction:
            # print("found matching p3")
            p3 = candidate

    return p3


def derive_LFR_direction(p1, p2, p3):
    """
    derive Left, Forward, Right directions
    input:
        p1 is the point two positions earlier
        p2 is the point one position earlier
        p3 is the current point
        both p1 and p2 are tuples of (x,y)

    return:
        left: 0,
        forward: 1,
        right: 2,
        error: -1
    """
    # first check for illegal moves
    # current point cannot be folded back to p1
    if p3 == p1:
        # print("ILLEGAL: folded back to p1")
        return -1
    # current point cannot stay put as p2
    if p3 == p2:
        # print("ILLEGAL: cannot stay put at p2")
        return -1
    # cases for cross product
    to_left_product = np.array([0, 0, 1], dtype=int)
    to_right_product = np.array([0, 0, -1], dtype=int)
    to_forward_product = np.array([0, 0, 0], dtype=int)
    #print("p1, p2, p3: ", p1, p2, p3)
    # derive the delta vector
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    #print("v1, v2: ", v1, v2)
    # first pad the v1 and v2 into 3D vector
    v1_3d = np.append(v1, [0])
    v2_3d = np.append(v2, [0])
    #print("v1_3d, v2_3d: ", v1_3d, v2_3d)
    cross_product = np.cross(v1_3d, v2_3d)
    #print("cross v1xv2: ", cross_product)
    if np.array_equal(cross_product, to_left_product):
        return LFR_map["left"]  # "Left"
    elif np.array_equal(cross_product, to_right_product):
        return LFR_map["right"]  # "Right"
    elif np.array_equal(cross_product, to_forward_product):
        return LFR_map["forward"]  # "Forward"
    else:
        return LFR_map["error"]  # "probably out of range moves"


def directionize_conf_seq(labelled_conf,
                          print_table=False):
    """
    input:
        labelled_conf:
            is a list of tuples of ((x,y), 'H|P')
            from label_conf_seq(conf_seq, conf_title)

        print_table: whether to
            print a table of X, Y, State, and Direction
    output:
        append column of Left, Forward, Right
        to the labelled_conf
        e.g.
        [
            ((0, 0), 'H', None, None),
            ((0, 1), 'H', 1, 'forward'),
            ((1, 1), 'H', 2, 'right'),
            ...
        ]
    """
    directionized_conf = []

    if print_table:
        print("X\tY\tState\tDirection(LFR)")
    for index, item in enumerate(labelled_conf):
        if index <= 1:
            # 0th and 1st monomer have no direction
            direction = None
        else:
            # compute the LFR direction for the move
            direction = derive_LFR_direction(
                            labelled_conf[index-2][0],
                            labelled_conf[index-1][0],
                            labelled_conf[index][0]
                        )

        directionized_conf.append(
            (
                item[0],
                item[1],
                direction,
                LFR_map.get(str(direction)),
            )
        )

        # print("directionized_conf is now:")
        # print(directionized_conf)

        if print_table:
            print('{}\t{}\t{}\t{}\t{}'.format(
                item[0][0],
                item[0][1],
                item[1],
                direction,
                LFR_map.get(str(direction)),
            ))

    # print("len of directionized_conf = ", len(directionized_conf))

    return directionized_conf


def plot_HPSandbox_conf(labelled_conf,
                        display_mode="draw",
                        pause_t=0.5,
                        save_fig=False,
                        save_path="",
                        score=2022,
                        optima_idx=0,
                        info={}):
    """
    input:
        labelled_conf:
            transformed file sequence of xy coords with state:
            ((x,y), 'H|P')
            e.g:
            [((0, 0), 'H'),
            ((0, 1), 'H'),...
            ((3, 1), 'P')]
        display_mode:
            draw vs show
        pause_t:
            seconds to display draw in plt.pause()
        save_fig:
            whether to save the pdf fig with seq name
            otherwise leaves a fig-live.pdf for review
    output:
        plot.show
    """

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=24)     # fontsize of the axes title
    plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=21)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=21)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=15)    # legend fontsize
    # plt.rc('figure', titlesize=48)  # fontsize of the figure title
    # plt.rcParams["figure.figsize"] = (8, 7)
    # plt.rcParams["font.family"] = "monospace"

    # print("+=+=+=+=+=+ plot_HPSandbox_conf -_-_-_-_-_-")
    x = [t[0][0] for t in labelled_conf]
    y = [t[0][1] for t in labelled_conf]
    str_seq = ''.join([t[1] for t in labelled_conf])
    assert len(str_seq) == info["chain_length"]
    H_seq = [t[0] for t in labelled_conf if t[1] == 'H']
    P_seq = [t[0] for t in labelled_conf if t[1] == 'P']

    # print("x: ", x)
    # print("y: ", y)
    # print("str_seq: ", str_seq)
    # print("H_seq: ", H_seq)
    # print("P_seq: ", P_seq)

    # Width, height in inches.
    fig_width = 5
    fig_height = 5
    # fontsize for legend
    fontsize = "xx-small"
    title_font = 12
    if len(str_seq) > 10:
        fig_width = 10
        fig_height = 10
        fontsize = "x-small"
        title_font = 15
    if len(str_seq) > 20:
        fig_width = 13
        fig_height = 13
        fontsize = "small"
        title_font = 18
    if len(str_seq) > 30:
        fig_width = 16
        fig_height = 16
        title_font = 21
    if len(str_seq) > 40:
        fig_width = 18
        fig_height = 18
        title_font = 24
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.subplots_adjust(top=0.9) # use a lower number to make more vertical space

    # set x and y limit and center origin
    max_xval = np.absolute(x).max()
    max_yval = np.absolute(y).max()
    ax.set_xlim(-max_xval-1, max_xval+1)
    ax.set_ylim(-max_yval-1, max_yval+1)

    # grid background
    ax.grid(linewidth=0.6, linestyle=':')

    # adjust plots with equal axis ratios
    #ax.axis('equal')
    ax.set_aspect('equal')  # , adjustable='box')

    # x and y axis tick at integer level
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # figure title
    # split the seq into chunks of chunk_size for the matplotlib title
    # NOTE: here the chunks are fixed sequence length not chain length
    chunks, chunk_size = info['seq_length'], 20
    # print(f"chunks={chunks}, chunk_size={chunk_size}")
    pad_len = info['seq_length'] - info['chain_length']

    # pad the str_seq
    str_seq = str_seq + '_'*pad_len
    seq_title_list = [
        str_seq[i:i+chunk_size]+"\n" for i in range(0, chunks, chunk_size)
    ]
    # print("seq_title_list = ", seq_title_list)
    seq_title_str = ''.join(seq_title_list)

    # do the same for padded action_str
    # pad the action_str
    action_str = ''.join(info["actions"])
    padded_action_str = action_str + '_'*pad_len
    action_title_list = [
        padded_action_str[i:i+chunk_size]+"\n" for i in range(0, chunks, chunk_size)
    ]
    # print("action_title_list = ", action_title_list)
    action_title_str = ''.join(action_title_list)

    title = "{}Actions=\n{}".format(
        seq_title_str,
        action_title_str,
    )
    # print("Title: ", title)
    ax.set_title(title, fontsize=title_font)

    # axis title
    ax.set_xlabel("x coord")
    ax.set_ylabel("y coord")

    # the HP plot consists of three layers

    # layer 1: backbone with solid line
    ax.plot(
        x, y,
        color='cornflowerblue',
        linewidth=4,
        label="backbone",
    )
    # layer 2: H as solid blue dots
    ax.plot(
        [h[0] for h in H_seq],
        [h[1] for h in H_seq],
        'o',
        markersize=14,
        label="H",
    )
    # layer 3: P as hollow orange dots
    ax.plot(
        [p[0] for p in P_seq],
        [p[1] for p in P_seq],
        'o',
        fillstyle='none',
        markersize=14,
        label="P",
    )

    # Show H-H bonds
    ## Compute all pair distances for the bases in the configuration
    coordinates = []
    for i in range(len(labelled_conf)):
        if labelled_conf[i][1] == 'H':
            coordinates.append(labelled_conf[i][0])
        else:
            coordinates.append((-1000, 1000)) #To get rid of P's
    distances = euclidean_distances(coordinates, coordinates)
    ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
    ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
    bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
    for (x,y) in zip(*bond_idx):
        xdata = [coordinates[x][0], coordinates[y][0]]
        ydata = [coordinates[x][1], coordinates[y][1]]
        backbone = mlines.Line2D(xdata, ydata, color = 'r', ls = ':', zorder = 1)
        ax.add_line(backbone)

    # show the legend
    # ax.legend(loc=0, markerscale=0.5, fontsize=fontsize)

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    if save_fig:
        # save the pdf fig with seq name
        plt.savefig(f"{save_path}/" +\
            f"{str_seq[:6]}_{action_str[:6]}_" +\
            f"{info['seq_length']}mer_" +\
            f"E{int(score)}_ID{optima_idx}.png",
            bbox_inches='tight',
        )

    if display_mode == "draw":
        # plt.draw()
        # plt.pause(pause_t)
        # save as pdf for review
        # can open the pdf in another window
        # the process is fast enough to get semi-live update rate
        # plt.savefig("fig-live.pdf", bbox_inches='tight')
        pass
    elif display_mode == "show":
        # show() is blocking, close window manually
        plt.show()

    # close the plt window
    plt.close()

def output_CNN(labelled_conf, N):
    # print("+=+=+=+=+=+ plot_HPSandbox_conf -_-_-_-_-_-")
    x = [t[0][0] for t in labelled_conf]
    y = [t[0][1] for t in labelled_conf]
    str_seq = '-'.join([t[1] for t in labelled_conf])
    H_seq = [t[0] for t in labelled_conf if t[1] == 'H']
    P_seq = [t[0] for t in labelled_conf if t[1] == 'P']

    # print("x: ", x)
    # print("y: ", y)
    # print("str_seq: ", str_seq)
    # print("H_seq: ", H_seq)
    # print("P_seq: ", P_seq)

    # Width, height in inches.
    fig_width = 5 * (N/20)
    fig_height = 5 * (N/20)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # set x and y limit and center origin
    # Sep30 new consistent plot for CNN
    # N is the complete chain length (not labeled_conf!)
    top_limit = N-0.5
    left_limit = -(N-1.5)
    bottom_limit = min(-(N-3.5), -0.5)
    right_limit = max(N-4.5, 0.5)
    # print(f"before crop, top={top_limit} left={left_limit} bot={bottom_limit} right={right_limit}")
    top_limit = round(top_limit/2) + 0.5
    left_limit = round(left_limit/2) - 0.5
    bottom_limit = (round(bottom_limit/2) - 0.5) if bottom_limit != -0.5 else bottom_limit
    right_limit = (round(right_limit/2) + 0.5) if right_limit != 0.5 else right_limit
    # print(f"after crop, top={top_limit} left={left_limit} bot={bottom_limit} right={right_limit}")
    ax.set_xlim(left_limit, right_limit)
    ax.set_ylim(bottom_limit, top_limit)

    # grid background
    # ax.grid(linewidth=0.6, linestyle=':')

    # adjust plots with equal axis ratios
    #ax.axis('equal')
    ax.set_aspect('equal')  # , adjustable='box')

    # https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    # the HP plot consists of three layers

    # layer 1: backbone with solid line
    ax.plot(
        x, y,
        color='cornflowerblue',
        linewidth=3,
    )
    # layer 2: H as solid blue dots
    ax.plot(
        [h[0] for h in H_seq],
        [h[1] for h in H_seq],
        'o',
        markersize=8,
    )
    # layer 3: P as hollow orange dots
    ax.plot(
        [p[0] for p in P_seq],
        [p[1] for p in P_seq],
        'o',
        # fillstyle='none',
        markersize=8,
    )

    # Show H-H bonds
    ## Compute all pair distances for the bases in the configuration
    coordinates = []
    for i in range(len(labelled_conf)):
        if labelled_conf[i][1] == 'H':
            coordinates.append(labelled_conf[i][0])
        else:
            coordinates.append((-1000, 1000)) #To get rid of P's
    distances = euclidean_distances(coordinates, coordinates)
    ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
    ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
    bond_idx = np.where(np.triu(distances, k=2) == 1.0)    
    for (x,y) in zip(*bond_idx):
        xdata = [coordinates[x][0], coordinates[y][0]]
        ydata = [coordinates[x][1], coordinates[y][1]]
        backbone = mlines.Line2D(xdata, ydata, color='r', ls='-', zorder=1, linewidth=1.5)
        ax.add_line(backbone)

    fig.tight_layout(pad=0)

    # import pyformulas as pf

    # canvas = np.zeros((480,640))
    # screen = pf.screen(canvas, 'Sinusoid')

    plt.ion()
    # based on canvas draw method
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    plt_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # print("plt_data.shape, plt_data.dtype = ", plt_data.shape, plt_data.dtype)
    # print(plt_data)
    # define the closest square root
    ns = np.ceil(np.sqrt(plt_data.shape[0]/3)).astype(int)
    # print("ns = ", ns)
    # print("fig.canvas.get_width_height() = ", fig.canvas.get_width_height())
    # print("fig.canvas.get_width_height()[::-1] = ", fig.canvas.get_width_height()[::-1])
    plt_data = plt_data.reshape((ns, ns) + (3,))
    # print("plt_data.shape, plt_data.dtype = ", plt_data.shape, plt_data.dtype)
    # print(plt_data)
    # screen.update(plt_data)

    # # based on savefig and buffer method
    # io_buf = io.BytesIO()
    # # Instead of saving the figure in png, one can use other format,
    # # like raw or rgba and skip the cv2 decoding step.
    # plt.savefig(io_buf, format='rgba', bbox_inches=0, pad_inches=0, dpi=30)
    # # plt.savefig(io_buf, format='raw')
    # io_buf.seek(0)
    # print("fig.bbox.bounds = ", fig.bbox.bounds)
    # buf_value = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    # print("buf_value.shape = ", buf_value.shape)
    # # define the closest square root
    # ns = np.ceil(np.sqrt(buf_value.shape[0])).astype(int)
    # print("ns = ", ns)
    # plt_data = np.reshape(buf_value,
    #                  newshape=(ns, ns, -1))
    # io_buf.close()
    

    # plt.savefig('CNN_plt_data.png', bbox_inches=0, pad_inches=0, dpi=30)
    # plt.show()

    # # close the plt window
    plt.close()

    return plt_data
