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
        
    
    def when_episode_starts(self, initial_observation, episode_index):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        return
        
    def when_action_needed(self, observation, reward):
        """
        returns an action from the action space
        """
        return
    
    def when_episode_ends(self, final_observation, reward, episode_index):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        return
    
    def when_should_clean(self):
        """
        only called once, and should save checkpoints and cleanup any logging info
        """
        return
