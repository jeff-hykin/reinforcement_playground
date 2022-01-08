from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from main.agent_builders.a2c.baselines_recreation.hacked_a2c import A2C
from main.agent_builders.a2c.baselines_optimizer import RMSpropTFLike
from main.agent_builders.a2c.baselines_recreation.auto_imitate import AutoImitator

import torch
from tools.debug import debug, ic
from tools.all_tools import Countdown, to_tensor, opencv_image_to_torch_image


# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=16 => 16 environments)
env = make_atari_env(
    'BreakoutNoFrameskip-v4',
    n_envs=16, # from atari optimized hyperparams
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
next_batch_triggered = Countdown(size=1000)
auto_imitator = AutoImitator(
    input_shape=(4,84,84),
    latent_shape=(512,),
    output_shape=(1,),
)

# 
# test
# 
observations = env.reset()
prev_actions = torch.ones((observations.shape[0],)) * 3
while True:
    batch_index += 1
    actions, _states = model.predict(observations)
    observations, rewards, dones, info = env.step(actions)
    # -1, 0, 1
    directions = torch.sign(to_tensor(prev_actions) - to_tensor(actions))
    auto_imitator.update_weights(
        batch_of_inputs=opencv_image_to_torch_image(observations),
        batch_of_ideal_outputs=directions,
        epoch_index=1,
        batch_index=batch_index,
    )
    if next_batch_triggered():
        print('batch_index = ', batch_index)
        torch.save(auto_imitator.state_dict(), "models.ignore/auto_imitator.model")
    prev_actions = actions
