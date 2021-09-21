import gym
from gym import spaces

# this is about as shallow of a wrapper as you can get
class Environment:
    def __init__(self, version=0, **config):
        # do stuff with the config
        self.config = config
        self._env = gym.make(f'Pendulum-v{version}')
    
    @property
    def observation_space(self):
        """
        Box(-8.0, 8.0, (3,), float32)
        """
        return self._env.observation_space
    
    @property
    def action_space(self):
        """
        Box(-2.0, 2.0, (1,), float32)
        """
        return self._env.action_space
    
    def reset(self):
        """
        should return the starting observation
        """
        return self._env.reset()
    
    def step(self, action):
        """
        observation, reward, episode_is_over, debugging_info = env.step(action)
        """
        return self._env.step(action)
    
    def close(self):
        """
        then cleanup after running all observations
        """
        return self._env.close()
        