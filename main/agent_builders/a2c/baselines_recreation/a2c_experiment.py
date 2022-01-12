from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from main.agent_builders.a2c.baselines_recreation.hacked_a2c import A2C
from main.agent_builders.a2c.baselines_optimizer import RMSpropTFLike
from main.agent_builders.a2c.baselines_recreation.auto_imitate import AutoImitator

import torch
from tools.debug import debug, ic
from tools.all_tools import Countdown, to_tensor, opencv_image_to_torch_image, to_pure
import tools.stat_tools as stat_tools
import silver_spectacle as ss


# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=16 => 16 environments)
env = make_atari_env(
    'BreakoutNoFrameskip-v4',
    n_envs=32, # from atari optimized hyperparams
    seed=0,
)
# Frame-stacking with 4 frames
env = VecFrameStack(
    env,
    n_stack=4, # from atari optimized hyperparams
)

# 
# load
# 
model = A2C(
    'CnnPolicy', # from atari optimized hyperparams
    env,
    verbose=1,
    ent_coef=0.01, # from atari optimized hyperparams
    vf_coef=0.25, # from atari optimized hyperparams
    policy_kwargs=dict(
        optimizer_class=RMSpropTFLike, # from atari optimized hyperparams
        optimizer_kwargs=dict(eps=1e-5), # from atari optimized hyperparams
    ),
)
model.load("baselines_10_000_000_model.ignore.zip")

#   
# setup imitator  
#   
batch_index = -1
next_group_triggered = Countdown(size=1000)
auto_imitator = AutoImitator(
    input_shape=(4,84,84),
    latent_shape=(512,),
    output_shape=(4,),
    path="models.ignore/auto_imitator_2.model",
)

# 
# test
# 
observations = env.reset()
while True:
    batch_index += 1
    observations_in_torch_form = opencv_image_to_torch_image(observations)
    
    agent_actions, _states = model.predict(observations)
    imitator_actions = auto_imitator.forward(observations_in_torch_form)
    observations, rewards, dones, info = env.step(agent_actions)
    auto_imitator.update_weights(
        batch_of_inputs=observations_in_torch_form,
        batch_of_ideal_outputs=agent_actions,
        epoch_index=1,
        batch_index=batch_index,
    )
    if next_group_triggered():
        print('batch_index = ', batch_index)
        auto_imitator.save()

ss.DisplayCard("quickLine", stat_tools.rolling_average(AutoImitator.logging.proportion_correct_at_index, 100))