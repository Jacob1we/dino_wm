from gym.envs.registration import register
import warnings

# Try to import MuJoCo-dependent environments (optional)
try:
    from .pointmaze import U_MAZE
    _HAS_MUJOCO = True
except Exception as e:
    _HAS_MUJOCO = False
    warnings.warn(f"MuJoCo not available, pointmaze environment disabled: {e}")

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

# point_maze nur registrieren wenn MuJoCo verfügbar
if _HAS_MUJOCO:
    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec':U_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

# Franka Cube Stack Environment (für Isaac Sim Integration)
register(
    id="franka_cube_stack",
    entry_point="env.franka_cube_stack:FrankaCubeStackWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)