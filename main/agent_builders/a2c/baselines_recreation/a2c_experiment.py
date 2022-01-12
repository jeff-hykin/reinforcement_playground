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
next_group_triggered = Countdown(size=250)
auto_imitator_01 = AutoImitator(
    learning_rate=0.01,
    input_shape=(4,84,84),
    latent_shape=(512,),
    output_shape=(4,),
    path="models.ignore/auto_imitator_01.model",
)
auto_imitator_001 = AutoImitator(
    learning_rate=0.001,
    input_shape=(4,84,84),
    latent_shape=(512,),
    output_shape=(4,),
    path="models.ignore/auto_imitator_001.model",
)
auto_imitator_0001 = AutoImitator(
    learning_rate=0.0001,
    input_shape=(4,84,84),
    latent_shape=(512,),
    output_shape=(4,),
    path="models.ignore/auto_imitator_0001.model",
)

# 
# train and record
# 
loss_card_01 = ss.DisplayCard("multiLine", [])
ss.DisplayCard("quickMarkdown", "#### Losses")
proportion_correct_card_01 = ss.DisplayCard("multiLine", [])
ss.DisplayCard("quickMarkdown", "#### Proportion Correct 01")
loss_card_001 = ss.DisplayCard("multiLine", [])
ss.DisplayCard("quickMarkdown", "#### Losses 001")
proportion_correct_card_001 = ss.DisplayCard("multiLine", [])
ss.DisplayCard("quickMarkdown", "#### Proportion Correct 001")
loss_card_0001 = ss.DisplayCard("multiLine", [])
ss.DisplayCard("quickMarkdown", "#### Losses 0001")
proportion_correct_card_0001 = ss.DisplayCard("multiLine", [])
ss.DisplayCard("quickMarkdown", "#### Proportion Correct 0001")
group_index = 0
observations = env.reset()
while True:
    batch_index += 1
    observations_in_torch_form = opencv_image_to_torch_image(observations)
    
    agent_actions, _states = model.predict(observations)
    observations, rewards, dones, info = env.step(agent_actions)
    auto_imitator_01.update_weights( batch_of_inputs=observations_in_torch_form, batch_of_ideal_outputs=agent_actions, epoch_index=1, batch_index=batch_index, )
    auto_imitator_001.update_weights( batch_of_inputs=observations_in_torch_form, batch_of_ideal_outputs=agent_actions, epoch_index=1, batch_index=batch_index, )
    auto_imitator_0001.update_weights( batch_of_inputs=observations_in_torch_form, batch_of_ideal_outputs=agent_actions, epoch_index=1, batch_index=batch_index, )
    if next_group_triggered():
        group_index += 1
        
        average_correct = to_tensor(auto_imitator_01.logging.proportion_correct_at_index[-next_group_triggered.size:]).mean()
        average_loss = to_tensor(auto_imitator_01.logging.loss_at_index[-next_group_triggered.size:]).mean()
        print(f'01 batch_index: {batch_index}, average_correct: {average_correct}, average_loss: {average_loss}')
        proportion_correct_card_01.send([group_index, average_correct])
        loss_card_01.send([group_index, average_loss])
        auto_imitator_01.save()
        
        average_correct = to_tensor(auto_imitator_001.logging.proportion_correct_at_index[-next_group_triggered.size:]).mean()
        average_loss = to_tensor(auto_imitator_001.logging.loss_at_index[-next_group_triggered.size:]).mean()
        print(f'001 batch_index: {batch_index}, average_correct: {average_correct}, average_loss: {average_loss}')
        proportion_correct_card_001.send([group_index, average_correct])
        loss_card_001.send([group_index, average_loss])
        auto_imitator_001.save()
        
        average_correct = to_tensor(auto_imitator_0001.logging.proportion_correct_at_index[-next_group_triggered.size:]).mean()
        average_loss = to_tensor(auto_imitator_0001.logging.loss_at_index[-next_group_triggered.size:]).mean()
        print(f'0001 batch_index: {batch_index}, average_correct: {average_correct}, average_loss: {average_loss}')
        proportion_correct_card_0001.send([group_index, average_correct])
        loss_card_0001.send([group_index, average_loss])
        auto_imitator_0001.save()
        
ss.DisplayCard("multiLine", stat_tools.rolling_average(auto_imitator.logging.proportion_correct_at_index, 100))