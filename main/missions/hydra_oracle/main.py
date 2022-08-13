import random
import time

import numpy as np
import gym
from gym import spaces
from stable_baselines3 import SAC

from missions.hydra_oracle.sac_exposed import SAC

from world_builders.fight_fire.world import World

world = World(
    grid_width=3,
    grid_height=3,
    visualize=False,
    fire_locations=[(-1,-1)],
    water_locations=[(0,0)],
)
env = world.Player()

# env = gym.make("Pendulum-v1")
# env = Env()
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100)


# class Network:
#     def __init__(self, prediction_shape, ):
#         pass
    

# class Agent:
#     def __init__(self, ):
#         self.body = BodyNetwork()
#         self.