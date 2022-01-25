from collections import defaultdict
from collections import defaultdict
from time import time as now
import functools
import math
import random

from gym.wrappers import AtariPreprocessing
from informative_iterator import ProgressBar
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from statistics import mean as average
from super_map import LazyDict, Map
from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
import torch


import tools.stat_tools as stat_tools
from tools.basics import product, flatten, to_pure, Countdown
from tools.debug import debug
from tools.pytorch_tools import layer_output_shapes, opencv_image_to_torch_image, to_tensor, init, forward, Sequential
from tools.frame_que import FrameQue
from tools.schedulers import AgentLearningRateScheduler
from tools.agent_recorder import AgentRecorder

from prefabs.baselines_optimizer import RMSpropTFLike
from prefabs.helpful_fitness_measures import trend_up, average

from agent_builders.a2c_baselines.main import Agent as A2C
from agent_builders.auto_imitator.main import Agent # as Imitator
from world_builders.atari.custom_preprocessing import preprocess


def default_mission(
        env_name="BreakoutNoFrameskip-v4",
        number_of_episodes=500,
        frame_buffer_size=4, # open ai defaults to 4 (VecFrameStack)
        frame_sample_rate=4,    # open ai defaults to 4, see: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        discount_factor=0.99,
        learning_rate=0.0007, # open ai defaults to 0.0007 for a2c
        path=None,
        random_proportion=0,
        episode_timeout=25,
    ):
    
    env = preprocess(
        env=gym.make(env_name),
        frame_buffer_size=frame_buffer_size,
        frame_sample_rate=frame_sample_rate,
    )
    
    # mr_bond = A2C.load("models.ignore/BreakoutNoFrameskip-v4.zip")
    mr_bond = Agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        should_print=False,
        random_proportion=random_proportion,
        path=path,
        # path=f"models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00020172481467972952.model",  # average reward 9.2 at random=0.4
        # path=f"models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00020898165099704166.model", # always chooses 0, gets 0 reward with random=0.8,0.5,0.1
        # path=f"models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000246800656.model", 
        # path=f"models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000221255917.model", # best one I think (0.009 randomness)
        # path=f"models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000206924927.model",
    )
    
    logger = LazyDict(
        should_log=Countdown(size=100),
    )
    
    mr_bond.when_mission_starts()
    for progress, episode_index in ProgressBar(number_of_episodes, disable_logging=True):
        mr_bond.observation = env.reset()
        mr_bond.reward = 0
        mr_bond.episode_is_over = False
        
        mr_bond.when_episode_starts(episode_index)
        start_time = now()
        timestep_index = -1
        while not mr_bond.episode_is_over:
            timestep_index += 1
            
            # check if episode is taking too long
            if episode_timeout < now() - start_time:
                return mr_bond
            
            mr_bond.when_timestep_starts(timestep_index)
            mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
            mr_bond.when_timestep_ends(timestep_index)
            
        mr_bond.when_episode_ends(episode_index)
    mr_bond.when_mission_ends()
    env.close()
    return mr_bond

def tune_hyperparams(number_of_episodes_per_trial=100_000, fitness_func=trend_up):
    import optuna
    # connect the trial-object to hyperparams and setup a measurement of fitness
    objective_func = lambda trial: fitness_func(
        default_mission(
            number_of_episodes=number_of_episodes_per_trial,
            discount_factor=trial.suggest_loguniform('discount_factor', 0.990, 0.991),
            learning_rate=trial.suggest_loguniform('learning_rate', 0.00070, 0.00071),
        ).logging.episode_rewards
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=50)
    return study

# 
# run
# 
random_rates = [ 0.5, 0.03, 0.1, 0.049, 0.009, 0.0001, ]
paths = [
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_4_7.255158601676353e-05.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_4_7.333068516151459e-05.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_4_8.230326041492979e-05.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_4_8.273254490838283e-05.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_4_9.323455721134604e-05.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_4_9.660588144057103e-05.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_4_9.933031240571859e-05.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.000190915141836022.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.000231290980530908.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.0001061637122626002.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.0001530834983455067.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.0001714488305872788.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.0002081996059583367.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.0002231532255232849.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.0002590805067421197.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00011059264201778339.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00012201569900076882.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00012568014538865712.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00013256096436012931.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00013414977467585322.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00015166186951154307.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00016320565888116884.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00017469580436010732.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00017975568487554305.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00018803185245693554.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00019058073773103553.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00019251253315966522.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00020172481467972952.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00020866622238544086.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00020898165099704166.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00023684711468376608.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00025208808783993674.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00028648679107851684.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_0.00028825892174588513.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_7.330169670225621e-05.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_8.675461183936687e-05.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_5_9.846282994344578e-05.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_6_0.000084407439.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_6_0.000113584572.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000074111241.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000075921072.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000080808598.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000085952679.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000087287003.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000088948328.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000092262921.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000093408395.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000094157877.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000096989272.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000097074061.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000098383606.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000100988997.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000102281871.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000104861100.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000106861278.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000108202428.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000108213181.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000108902888.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000109039653.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000111631595.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000112464959.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000114267607.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000114996759.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000115118452.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000115656138.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000117193527.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000120552473.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000123065215.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000124264122.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000128483367.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000128858096.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000130179438.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000132100360.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000133176788.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000134853188.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000138390956.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000138707078.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000139845239.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000142148532.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000145002650.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000147158579.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000149861495.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000151478996.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000152685371.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000153511651.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000157044798.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000157184687.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000157653836.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000159121446.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000159206199.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000160683460.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000161672050.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000163811219.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000164009785.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000164273321.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000164911138.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000168394876.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000168654194.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000170264552.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000171426049.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000174712371.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000175127593.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000176037294.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000178195535.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000178791579.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000179133440.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000179588275.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000180808054.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000181755364.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000182594345.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000184523995.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000184719039.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000186450345.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000187735534.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000190241887.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000191715671.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000192749715.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000192880004.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000194966399.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000195333990.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000198926836.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000201998968.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000203522155.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000203702994.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000203981199.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000204270238.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000206924927.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000211810528.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000212590315.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000214681865.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000220707832.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000221255917.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000225778823.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000229174533.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000237143204.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000246800656.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000250362957.model",
    "models.ignore/auto_imitator_hacked_compressed_preprocessing_7_0.000265237521.model",
]
random.shuffle(paths)
for each_path in paths:
    name = each_path.replace("models.ignore/auto_imitator_hacked_compressed_preprocessing_", "")
    for each_random_rate in random_rates:
        try:
            agent = default_mission(
                number_of_episodes=30,
                random_proportion=each_random_rate,
                path=each_path,
            )
            print(f"{name}: episodes: {len(agent.logging.episode_rewards)} random: {each_random_rate} reward:{agent.logging.across_episodes.average_reward} actions: {agent.action_from_given_observation}")
        except Exception as error:
            print('error = ', error)
        