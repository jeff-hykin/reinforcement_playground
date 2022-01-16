import torch
# import silver_spectacle as ss
from statistics import mean as average
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from tools.debug import debug, ic
from tools.all_tools import Countdown, to_tensor, opencv_image_to_torch_image, to_pure
import tools.stat_tools as stat_tools

from world_builders.atari.baselines_vectorized import Environment
from agent_builders.baselines_a2c.main import Agent
from prefabs.baselines_optimizer import RMSpropTFLike
from prefabs.helpful_fitness_measures import average

env = Environment(
    name='BreakoutNoFrameskip-v4',
    n_envs=1, # from atari optimized hyperparams
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
# eval
# 
number_of_episodes = 100


#   
# mission
#   
mr_bond.when_mission_starts()
for episode_index in range(number_of_episodes):
    
    mr_bond.observation = env.reset()
    mr_bond.reward = 0
    mr_bond.episode_is_over = [False]
    
    mr_bond.when_episode_starts(episode_index)
    timestep_index = -1
    while not mr_bond.episode_is_over[0]:
        timestep_index += 1
        
        mr_bond.when_timestep_starts(timestep_index)
        mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
        mr_bond.when_timestep_ends(timestep_index)
            
    mr_bond.when_episode_ends(episode_index)
    
    print('average rewards = ', average(mr_bond.log.episode_rewards))
    
mr_bond.when_mission_ends()
env.close()        