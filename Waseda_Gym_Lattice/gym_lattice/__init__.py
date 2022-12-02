from gym.envs.registration import register

# NOTE: for gym envs IDs:
# Currently all IDs must be of the form:
# ^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$.
# register(
#     # v0 is before the Waterloo Pull Request
#     # v1 is after 2020 Jul
#     id='Lattice2D-miranda2020Jul-v1',
#     entry_point='gym_lattice.envs:Lattice2DEnv',
# )

# register(
#     id='Lattice2D-4actionStateEnv-v0',
#     entry_point='gym_lattice.envs:FourActionStateEnv',
# )

register(
    id='Lattice2D-3actionStateEnv-v0',
    entry_point='gym_lattice.envs:ThreeActionStateEnv',
)
