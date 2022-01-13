import torch
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
mr_bond.load("models.ignore/BreakoutNoFrameskip-v4.zip")

#   
# setup database
#   
database = AgentRecorder(save_to="resources/datasets.ignore/atari/baselines_pretrained@vectorized_breakout")

# 
# run env
# 
batch_index = -1
group_index = -1
observations = env.reset()
mr_bond.when_mission_starts()
mr_bond.when_episode_starts(0)
next_group_triggered = Countdown(size=1000)
while True:
    batch_index += 1
    observations_in_torch_form = opencv_image_to_torch_image(observations)
    
    mr_bond.observation = observations
    mr_bond.when_timestep_starts(batch_index)
    actions = mr_bond.action
    observations, rewards, dones, info = env.step(actions)
    mr_bond.when_timestep_ends(batch_index)
    
    # save
    for each_observation, each_action in zip(observations, actions):
        database.save(each_observation, each_action)
    
    if batch_index < next_group_triggered.size:
        print(f'checkin: batch_index: {batch_index}')
        
    if next_group_triggered():
        group_index += 1
        print(f'batch_index: {batch_index}, ')
