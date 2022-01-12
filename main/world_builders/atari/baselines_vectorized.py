from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def Environment(name='BreakoutNoFrameskip-v4', *, n_envs=16, seed=0, n_stack=4):
    return VecFrameStack(
        make_atari_env(
            name,
            n_envs=n_envs, # from atari optimized hyperparams
            seed=seed,
        ),
        n_stack=n_stack, # 4 frames, from atari optimized hyperparams
    )

