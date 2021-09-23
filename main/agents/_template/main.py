import torch
import torch.nn as nn
import torch.nn.functional as F

# local 
from tools.all_tools import PATHS

class Agent:
    def __init__(self, action_space=None, observation_space=None, **config):
        """
        arguments:
            action_space: is a gym space (from gym import spaces)
        """
        # args
        self.action_space = action_space
        self.observation_space = observation_space
        self.config = config
        
        # required poperties
        self.observation = None
        self.aciton = None
        self.wants_to_quit = False
        
        # logging tool
        self.log = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
    
    def get_reward(self, observation, action=None):
        """
        should return a reward value based on what the agent can observe
        """
        return 0
    
    def when_episode_starts(self, episode_index):
        """
            anything that you might want to be run on a per-episode basis
        """
        return
        
    def when_time_passes(self):
        """
            check self.observation to see what the agent sees
            change self.action to act inside the environment
        """
    
    def when_episode_ends(self, episode_index):
        """
            anything that you might want to be run on a per-episode basis
        """
        return
    
    def when_should_clean(self):
        """
            only called once, and should save checkpoints and cleanup any logging info
        """
        return