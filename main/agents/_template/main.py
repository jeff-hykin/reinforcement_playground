import torch
import torch.nn as nn
import torch.nn.functional as F

# local 
from tools.all_tools import PATHS

class Agent:
    def __init__(self, action_space=None, **config):
        """
        arguments:
            action_space: is a gym space (from gym import spaces)
        """
        self.config = config
        self.action_space = action_space
        self.wants_to_quit = False
        self.show = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
        
    
    def on_episode_start(self, initial_observation):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        return
        
    def decide(observation, reward, is_last_timestep):
        """
        returns an action from the action space
        """
        return
    
    def on_clean_up(self):
        """
        only called once, and should save checkpoints and cleanup any logging info
        """
        return
