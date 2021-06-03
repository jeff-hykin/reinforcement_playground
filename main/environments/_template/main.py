from gym import spaces

class Environment:
    def __init__(self, **config):
        # do stuff with the config
        self.config = config
    
    @property
    def observation_space(self):
        return None
        # example1:
        # Set with 8 elements {0, 1, 2, ..., 7}
        space = spaces.Discrete(8) 
        return space
    
    @property
    def action_space(self):
        return None
        # example1:
        # Set with 8 elements {0, 1, 2, ..., 7}
        space = spaces.Discrete(8) 
        return space
    
    def reset(self):
        """
        should return the starting observation
        """
        return None
    
    def step(self, action):
        """
        observation, reward, episode_is_over, debugging_info = env.step(action)
        """
        return None, None, None, None
    
    def close(self):
        """
        then cleanup after running all observations
        """
        return
        