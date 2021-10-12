import torch
import torch.nn as nn
import torch.nn.functional as F

# local 
from tools.all_tools import PATHS
from tools.reinverse import ConnectBody

@ConnectBody
class Brain:
    def __init__(self, body):
        self.body = body # call it self.whatever_you_want, just need a body argument
        self.action_space = self.body.action_space
        self.observation_space = self.body.observation_space
    
    @ConnectBody.when_episode_starts
    def your_method1(self, episode_index):
        self.body.get_observation()
        self.body.get_reward()
        self.body.take_action(self.body.action_space.sample())
        