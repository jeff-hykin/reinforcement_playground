# import random
# import time

# import numpy as np
# import gym
# from gym import spaces
# from stable_baselines3 import SAC
# from stable_baselines3 import A2C

# from missions.hydra_oracle.sac_exposed import SAC
# from missions.hydra_oracle.a2c_exposed import SAC

# from world_builders.fight_fire.world import World
# from world_builders.atari.world import World

# world = World(
#     # grid_width=3,
#     # grid_height=3,
#     # visualize=False,
#     # fire_locations=[(-1,-1)],
#     # water_locations=[(0,0)],
# )
# env = world.Player()

# # env = gym.make("Pendulum-v1")
# # env = Env()
# model = A2C('CnnPolicy', env, verbose=1)
# model.learn(total_timesteps=250)

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C

from missions.hydra_oracle.a2c_exposed import A2C
from missions.hydra_oracle.policies import ActorCriticCnnPolicy

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4) # Frame-stacking with 4 frames

model = A2C(ActorCriticCnnPolicy, env, verbose=1)
model.learn(total_timesteps=25_000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(f'''rewards = {rewards}''')
    if dones[0]:
        break

# class Network(nn.Module):
#     @init.to_device()
#     def __init__(self, *, input_size, output_size, number_of_layers, learning_rate=0.1):
#         super(Network, self).__init__()
        
#         # 
#         # body
#         # 
#         if True:
#             self.body = Sequential()
#             # FIXME: figure out how to pick network shape
#             self.body.add_module('l_1', nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4))
#             self.body.add_module('l_2', nn.ReLU()                                             )
#             self.body.add_module('l_3', nn.Conv2d(32, 64, kernel_size=4, stride=2),           )
#             self.body.add_module('l_4', nn.ReLU(),                                            )
#             self.body.add_module('l_5', nn.Conv2d(64, 64, kernel_size=3, stride=1),           )
#             self.body.add_module('l_6', nn.ReLU()                                             )
        
#         # 
#         # reward prediction head
#         # 
#             # TODO: network design
#             # TODO: loss function
        
#         # 
#         # state prediction head
#         # 
#             # TODO: network design
#             # TODO: loss function
        
#         # 
#         # action distribution
#         # 
#             # TODO: network design
#             # TODO: loss function
        
#         mse_loss = nn.MSELoss()
#         self.loss_function = lambda current_output, ideal_output: mse_loss(current_output, ideal_output)
#         self.optimizer = self.get_optimizer(learning_rate)
    
#     def get_optimizer(self, learning_rate):
#         return optim.SGD(self.parameters(), lr=learning_rate)
    
#     def predict(self, input_batch):
#         with torch.no_grad():
#             return self.forward(input_batch)
        
#     def pipeline(self):
#         return self.body.lstm.pipeline()

# class Agent:
#     def __init__(self, ):
#         pass
    
#     def when_mission_starts(self):
#         pass
    
#     def when_episode_starts(self):
#         pass
    
#     def when_timestep_starts(self):
#         print([ each for each in self.reactions ])
#         pass
    
#     def when_timestep_ends(self):
#         pass
    
#     def when_episode_ends(self):
#         pass
    
#     def when_mission_ends(self):
#         pass