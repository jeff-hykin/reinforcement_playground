import torch
import silver_spectacle as ss
from statistics import mean as average
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from tools.debug import debug, ic
from tools.all_tools import Countdown, to_tensor, opencv_image_to_torch_image, to_pure
import tools.stat_tools as stat_tools
from tools.agent_recorder import AgentRecorder

from world_builders.atari.baselines_vectorized import Environment
from agent_builders.baselines_a2c.main import Agent
from prefabs.baselines_optimizer import RMSpropTFLike
from prefabs.auto_imitator import AutoImitator

env = Environment(
    name='BreakoutNoFrameskip-v4',
    n_envs=32, # from atari optimized hyperparams
    seed=0,
    n_stack=4, # 4 frames, from atari optimized hyperparams
)

# 
# load
# 
mr_bond = Agent(
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
# mr_bond.load("models.ignore/baselines_10_000_000.model")
mr_bond.load("models.ignore/BreakoutNoFrameskip-v4.zip")

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
    path="models.ignore/auto_imitator_4.model",
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
mr_bond.when_mission_starts()
mr_bond.when_episode_starts(0)
while True:
    batch_index += 1
    observations_in_torch_form = opencv_image_to_torch_image(observations)
    
    mr_bond.observation = observations
    mr_bond.when_timestep_starts(batch_index)
    observations, rewards, dones, info = env.step(mr_bond.action)
    auto_imitator.update_weights(
        batch_of_inputs=observations_in_torch_form,
        batch_of_ideal_outputs=mr_bond.action,
        epoch_index=1,
        batch_index=batch_index,
    )
    mr_bond.when_timestep_ends(batch_index)
    
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