# Import gym modules
import sys
from collections import OrderedDict

import gym
from gym import (spaces, utils, logger)
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# from gym_lattice.envs.lattice2d_env_miranda_repo_2020Jul import Lattice2DEnv

# from gym_lattice.envs.utils.obs_quaternary_to_base10 import obs_quaternary_to_base10
# from gym_lattice.envs.utils.obs_quaternary_to_3N2 import obs_quaternary_to_3N2

# import the hpsandbox util functions
sys.path.append('../code')
from hpsandbox_util import plot_HPSandbox_conf#, output_CNN
from hpsandbox_util import move_LFR_direction


# Over-write the miranda's default action to string dict
ACTION_TO_STR = {
    0: 'L',
    1: 'F',
    2: 'R'}


class ThreeActionStateEnv(gym.Env):
    """
    Inherits from Miranda's Lattice2DEnv (2D lattice environment)

    Three actions (Left, Forward, Right)

    Nov 25 2021 change to always DO NOT allow collision
    """

    def __init__(self,
                 seq,
                 obs_output_mode="tuple",
                ):
        """
        Parameters
        ----------
        seq : str, must only consist of 'H' or 'P'
            Sequence containing the polymer chain.
        obs_output_mode : str, whether to output the observations
            "tuple" --> as quaternary tuple of actions,
            "index_quaternary" --> return the int of the decimal index
            "index_3N2" --> return int of the 3N2 index
        # for british museum use the state_E as reward, no need partial_reward
        """
        self.seq = seq.upper()
        self.obs_output_mode = obs_output_mode

        # Initialize values
        self.reset()

        # customized 3actionStateEnv
        print("=================Three Action State Env===============\n")

        # Three Actions the seq must be longer than 2
        if len(self.seq) <= 2:
            return
        # action_space is LFR actions
        self.action_space = spaces.Discrete(3)
        # for obs space, [0] means no action taken yet
        # [1,2,3] matches [L, F, R] actions
        self.observation_space = spaces.Box(low=0, high=3,
                                            # quaternary tuple len is N-2
                                            shape=(len(self.seq)-2,),
                                            dtype=int)
        # NOTE: obs space of [2,2,3] means actions taken: F, F, R

        # first_turn_left --> whether the first turn is a left
        # after reset, this is default to False
        self.first_turn_left = False

        # print attributes and states for sanity check
        print("ThreeActionStateEnv init with attributes:")
        print("self.seq = ", self.seq)
        print("len(self.seq) = ", len(self.seq))
        print("self.obs_output_mode = ", self.obs_output_mode)

        print("self.state = ", self.state)
        print("self.actions = ", self.actions)

        print("self.action_space:")
        print(self.action_space)
        print("self.observation_space:")
        print(self.observation_space)
        print("self.observation_space.high, low:")
        print(self.observation_space.high)
        print(self.observation_space.low)
        print("self.observation_space.shape:")
        print(self.observation_space.shape)
        print("self.observation_space.dtype, self.action_space.dtype")
        print(self.observation_space.dtype, self.action_space.dtype)

        print("self.first_turn_left = ", self.first_turn_left)


    def step(self, action):
        """
        Overload the parent class' step()
        """
        # print("\n****************************************")
        # print("**************   NEW STEP  ***************")
        # print("****************************************\n")

        # print("\n***********step's action*********")
        # print(action, ["Left", "Forward", "Right"][action])

        # print("\n***********step's state*********")
        # print(self.state)

        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))

        # print("self.first_turn_left = ", self.first_turn_left)
        # if action is a turn, check whether first turn is already left
        # action: 0,1,2 int
        if (action != 1) and (self.first_turn_left is False):
            # detect whether first turn is left (action 0) or right (action 2)
            if action == 2:
                action = 0
            # print("Converted action = ", action)
            self.first_turn_left = True
            # print("-- after conversion, self.first_turn_left = ", self.first_turn_left)

        self.last_action = action
        is_trapped = False # Trap signal

        # Obtain coordinate of previous polymer
        # OrderedDict.keys() gives the coords
        # p1 is the point two positions earlier
        # p2 is the point one position earlier
        # both p1 and p2 are tuples of (x,y)
        p2 = list(self.state.keys())[-1]
        p1 = list(self.state.keys())[-2]
        # p3 is to-be-moved point == next_move
        next_move = move_LFR_direction(
            p1=p1,
            p2=p2,
            move_direction=action,
        )
        # print("\n***********step's next_move*********")
        # print(next_move)
        # Detects for collision or traps in the given coordinate
        idx = len(self.state)
        if next_move in self.state:
            # print("next_move in self.state --> collision!")
            # Default does not allow collisions,
            # ie, step() will retry until the next_move is not colliding
            # send obs is None as signal for retry
            return (None, None, False, {})
        else:
            # only append valid actions to action chain
            self.actions.append(action)
            try:
                self.state.update({next_move : self.seq[idx]})
            except IndexError:
                logger.error('All molecules have been placed! Nothing can be added to the protein chain.')
                raise

            # NOTE: agent is only trapped WHEN THERE ARE STILL STEPS TO BE DONE!
            if len(self.state) < len(self.seq):
                if set(self._get_adjacent_coords(next_move).values()).issubset(self.state.keys()):
                    # logger.warn('Your agent was trapped! Ending the episode.')
                    is_trapped = True

        # Set-up return values
        obs = self.observe()
        # print("\n***********step's obs*********")
        # print(obs)

        self.done = True if (len(self.state) == len(self.seq) or is_trapped) else False
        reward = self._compute_reward()
        info = {
            'chain_length' : len(self.state),
            'seq_length'   : len(self.seq),
            'actions'      : [ACTION_TO_STR[i] for i in self.actions],
            'is_trapped'   : is_trapped,
            'state_chain'  : self.state,
            "first_turn_left": self.first_turn_left,
        }

        return (obs, reward, self.done, info)


    def observe(self):
        """
        convert self.actions list to a tuple for base-4 system
        0123 from actions (each action+1) --> 1234
        The tuple returned is a base-4 quaternary number tuple:
            0 = no action taken
            1 = Left
            2 = Forward
            3 = Right
        (3,2,2,0) = (R,F,F) for seq HPHPH

        output:
            NOTE: for NN compatibility, tuple is in fact np.array
            either tuple or int, depending on
            the self.obs_output_mode
            "tuple" --> return the quaternary tuple
            "index_quaternary" --> return the int of the decimal index
            "index_3N2" --> return int of the 3N2 index
        """
        # self.actions is list of 012 integers
        action_chain = self.actions

        # native obs space is [0,0,0,...,0]
        # len of quaternary tuple is N-2
        native_obs = np.zeros(shape=(len(self.seq)-2,), dtype=int)

        # transfer the actions in action chain to
        # the obs array (each action+1)
        for i, item in enumerate(action_chain):
            native_obs[i] = item+1

        # for NN input, preserve the np array instead of tuple
        quaternary_tuple = native_obs

        # print("self.obs_output_mode = ", self.obs_output_mode)

        if self.obs_output_mode == "tuple":
            # in fact is a numpy array not a tuple for NN
            return quaternary_tuple
        # elif self.obs_output_mode == "index_quaternary":
        #     return obs_quaternary_to_base10(quaternary_tuple)
        # elif self.obs_output_mode == "index_3N2":
        #     return obs_quaternary_to_3N2(quaternary_tuple)


    def reset(self):
        """
        Inherits and overload parent class
        Resets the environment
        """
        self.actions = []

        self.last_action = None
        self.prev_reward = 0

        # customized reset
        # in miranda Jul2020 baseEnv and in 4actionState,
        # the initial polymer is placed at origin
        # for 3actionState, place the next polyer at (0,1)
        self.state = OrderedDict(
            {
                (0, 0): self.seq[0],
                (0, 1): self.seq[1],
            }
        )
        self.done = len(self.seq) == 2
        obs = self.observe()
        # reset the first_turn_left for a new episode
        self.first_turn_left = False

        return obs


    def render(self, mode='human', display_mode="draw",
               pause_t=0.0, save_fig=False, save_path="",
               score=2022, optima_idx=0):
        """
        Use matplotlib to plot the grid
        and the chain
        """
        # print("\n^^^^^^ Render called ^^^^^^")

        # outfile = StringIO() if mode == 'ansi' else sys.stdout

        # print("render self.state: ", self.state)
        # print("list(self.state): ", list(self.state.items()))
        if mode == "human":
            # matplotlib plot the conf
            plot_HPSandbox_conf(
                list(self.state.items()),
                display_mode=display_mode,
                pause_t=pause_t,
                save_fig=save_fig,
                save_path=save_path,
                score=score,
                optima_idx=optima_idx,
                info={
                    'chain_length' : len(self.state),
                    'seq_length'   : len(self.seq),
                    'actions'      : [ACTION_TO_STR[i] for i in self.actions],
                },
            )
        # elif mode == "CNN":
        #     # print("mode is CNN!")
        #     plt_data = output_CNN(
        #         list(self.state.items()),
        #         len(self.seq)
        #     )
        #     return plt_data

        # Provide prompt for last action
        # if self.last_action is not None:
        #     outfile.write("Last Action:  ({})\n".format(["Left", "Forward", "Right"][self.last_action]))
        # else:
        #     outfile.write("\n")

        # if mode != 'human':
        #     return outfile

        # print("****** Render Done ******\n")


    def _get_adjacent_coords(self, coords):
        """Obtains all adjacent coordinates of the current position

        Parameters
        ----------
        coords : 2-tuple
            Coordinates (X-y) of the current position

        Returns
        -------
        dictionary
            All adjacent coordinates
        """
        x, y = coords
        adjacent_coords = {
            0 : (x - 1, y),
            1 : (x, y - 1),
            2 : (x, y + 1),
            3 : (x + 1, y),
        }

        return adjacent_coords


    def _compute_reward(self):
        # new Sep19 reward in tuple (state_E, step_E, reward)
        curr_reward = self._compute_free_energy(self.state)
        state_E = curr_reward
        step_E = curr_reward - self.prev_reward
        self.prev_reward = curr_reward
        reward = curr_reward if self.done else 0
        # NOTE: let the RL algo decide how to use the state_E and penalty_per_node Sep23
        # if is_trapped:
        #     # punish trapped episode by clear-out to be -5 energy
        #     state_E, step_E, reward = 5, 5, 5
        return (-state_E, -step_E, -reward)


    def _compute_free_energy(self, chain):
        """Computes the Gibbs free energy given the lattice's state

        The free energy is only computed at the end of each episode. This
        follow the same energy function given by Dill et. al.
        [dill1989lattice]_

        Recall that the goal is to find the configuration with the lowest
        energy.

        .. [dill1989lattice] Lau, K.F., Dill, K.A.: A lattice statistical
        mechanics model of the conformational and se quence spaces of proteins.
        Marcromolecules 22(10), 3986–3997 (1989)

        Parameters
        ----------
        chain : OrderedDict
            Current chain in the lattice

        Returns
        -------
        int
            Computed free energy
        """
        # alternative way to compute number of HH bonds
        use_triu_distance_mat = True
        if use_triu_distance_mat:
            dictlist = list(chain.items())
            coordinates = []
            for i in range(len(dictlist)):
                if dictlist[i][1] == 'H':
                    coordinates.append(dictlist[i][0])
                else:
                    coordinates.append((-1000, 1000)) #To get rid of P's
            # print("coordinates matrix = ", coordinates)
            distances = euclidean_distances(coordinates, coordinates)
            # print("distances matrix = ", distances)
            ## We can extract the H-bonded pairs by looking at the upper-triangular (triu)
            ## distance matrix, and taking those = 1, but ignore immediate neighbors (k=2).
            bond_idx = np.where(np.triu(distances, k=2) == 1.0)
            """
            bond_idx is a tuple
            for m HH bonds ->
            (array([x1, x2, ..., x_m]), array([y1, y2, ..., y_m]))
            """
            # print("bond_idx = ", bond_idx)
            # print("len(bond_idx[0]) = ", len(bond_idx[0]))
            reward = -1 * len(bond_idx[0])
            return int(reward)


    def seed(self, seed=None):
        """
        seed the gym env
        NOTE: this umbrella seed() will seed the end globally
        it will seed:
            action_space
            np random (for uniform to_exploit)
        """
        self.np_random, seed = utils.seeding.np_random(seed)

        # NOTE: spaces sample use separate random number generator
        # that lives in gym.spaces.prng. If you want action / observation
        # space to sample deterministically you will need to seed separately
        self.action_space.seed(seed)
        # NOTE: agent also uses randomness, need to seed that separately
        np.random.seed(seed)

        return [seed]
