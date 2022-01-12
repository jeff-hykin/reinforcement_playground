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
from statistics import mean as average


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
    learning_rate=0.001,
    input_shape=(4,84,84),
    latent_shape=(512,),
    output_shape=(4,),
    path="models.ignore/auto_imitator_3.model",
)

# 
# train and record
# 
loss_card = ss.DisplayCard("quickLine", [])
ss.DisplayCard("quickMarkdown", "#### Losses")
proportion_correct_card = ss.DisplayCard("quickLine", [])
ss.DisplayCard("quickMarkdown", "#### Proportion Correct")

group_index = 0
observations = env.reset()
while True:
    batch_index += 1
    observations_in_torch_form = opencv_image_to_torch_image(observations)
    
    agent_actions, _states = model.predict(observations)
    observations, rewards, dones, info = env.step(agent_actions)
    auto_imitator.update_weights(
        batch_of_inputs=observations_in_torch_form,
        batch_of_ideal_outputs=agent_actions,
        epoch_index=1,
        batch_index=batch_index,
    )
    
    if batch_index < next_group_triggered.size:
        print(f'checkin: batch_index: {batch_index}')
        
    if next_group_triggered():
        group_index += 1
        average_correct = to_tensor(auto_imitator.logging.proportion_correct_at_index[-next_group_triggered.size:]).mean()
        average_loss = to_tensor(auto_imitator.logging.loss_at_index[-next_group_triggered.size:]).mean()
        print(f'batch_index: {batch_index}, average_correct: {average_correct}, average_loss: {average_loss}')
        loss_card.send([group_index, average_loss])
        proportion_correct_card.send([group_index, average_correct])
        auto_imitator.save()

ss.DisplayCard("quickLine", stat_tools.rolling_average(auto_imitator.logging.proportion_correct_at_index, 100))