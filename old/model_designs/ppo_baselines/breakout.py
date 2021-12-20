# from old.environments.atari.main import Environment
from old.model_designs.ppo_baselines.hacked_atari import Environment

env = Environment(
    game="breakout",
    frameskip=0,
    repeat_action_probability=0.0,
)