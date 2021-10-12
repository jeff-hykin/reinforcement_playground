# packages
import gym
from gym import spaces
import numpy as np

# local
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from environments.template.main import Environment as BaseEnvironment

# just a class wrapper around the unity wrapper
class Environment(BaseEnvironment):
    default_config = {
        "min_throttle": 0.45,
        "max_throttle":  0.6,
        "max_steering_diff": 0.15,
        "jerk_reward_weight": 0.0,
        "max_steering": 1, # min_steering is negative of the max
        "steering_gain": 1,
        "steering_bias": 0,
    }
    
    def __init__(self, **config):
        self.config = config
        self.env = UnityToGymWrapper(
            UnityEnvironment(),
            allow_multiple_obs=True, # not exactly sure what this does,
        )
    
    @property
    def observation_space(self, action):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, self.z_size),
            dtype=np.float32,
        )
        
    @property
    def action_space(self, action):
        return spaces.Box(
            low=np.array([-self.config["max_steering"], -1]),
            high=np.array([self.config["max_steering"], 1]),
            dtype=np.float32,
        )
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)