class Skeleton:
    observation = None     # external world will change this
    reward = None          # external world will change this
    action = None          # extrenal world will read this
    episode_is_over = None # extrenal world will change this
    
    def __init__(self, observation_space, action_space, **config):
        self.observation = None     # external world will change this
        self.reward = None          # external world will change this
        self.action = None          # extrenal world will read this
        self.episode_is_over = None # extrenal world will change this

    def when_mission_starts(self, mission_index=0):
        pass
    
    def when_episode_starts(self, episode_index):
        pass
    
    def when_timestep_starts(self, timestep_index):
        # implement: self.action = something
        pass
    
    def when_timestep_ends(self, timestep_index):
        pass
    
    def when_episode_ends(self, episode_index):
        pass
    
    def when_mission_ends(self, mission_index=0):
        pass
    
    def update_weights(self):
        pass
    