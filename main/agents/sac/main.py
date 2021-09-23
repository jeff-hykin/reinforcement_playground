import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# local 
from tools.all_tools import PATHS
from tools.reality_maker import MinimalAgent

from spinup.algos.pytorch.sac.sac import sac


class Agent(MinimalAgent):
    def __init__(self, body_type, **config):
        super(Agent, self).__init__(body_type)
        # save config for later
        self.config = config
    
    def when_body_is_ready(self):
        # wrapper env
        agent = self
        class DummyEnv(gym.Env):
            @property
            def action_space(self): return agent.body.action_space
            @property
            def observation_space(self): return agent.body.observation_space
            @property
            def step(self, action): return None, None, None, None
            @property
            def close(self): pass
        
        self.spinup_sac = sac(
            env_fn=lambda: DummyEnv(),
            **self.config,
        )
        
    def when_campaign_starts(self):
        pass
        
    def when_episode_starts(self, episode_index):
        # FIXME: init stuff
        
    def when_time_passes(self):
        # FIXME: get new action
        self.spinup_sac
        
    def when_episode_ends(self, episode_index):
        # FIXME: update the state, and update the 
    
    def when_campaign_ends(self):
        pass