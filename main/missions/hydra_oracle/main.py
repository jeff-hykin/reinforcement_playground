import random
import time

import numpy as np
import gym
from gym import spaces
from stable_baselines3 import SAC

from world_builders.fight_fire.world import World

# class Env(gym.Env):
#     def __init__(self):
#         super(Env, self).__init__()
#        
#         # Define a 2-D observation space
#         self.observation_shape = (1,1)
#         self.observation_space = spaces.Box(low=np.zeros(self.observation_shape), high=np.ones(self.observation_shape), dtype=np.float16)
#        
#         # Define an action space ranging from 0 to 4
#         self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]) )
#    
#     def reset(self):
#         return np.array( random.randrange(0,1) )
#    
#     def step(self, action):
#         print(f'''action = {action}''')
#         reward = 0
#         done = False
#         next_state = np.array( random.randrange(0,1) )
#         debugging_info = {}
#         return next_state, reward, done, debugging_info

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