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

def rl_basics(init_function):
    """
        creates:
            self.timestep
            self.prev_timestep
            self.episode
            self.per_episode
            self.episodes
            self.prev_observation
            self.prev_observation_response
            self.action_frequency    (if self.actions)
    """
    from super_map import LazyDict
    from tools.basics import sort_keys, randomly_pick_from
    
    def wrapper(self, *args, **kwargs):
        output = init_function(self, *args, **kwargs)
        
        # 
        # action frequency
        # 
        def update_action_frequency():
            pass
        if hasattr(self, "actions") and type(self.actions) != type(None):
            self.action_frequency = LazyDict({ each:0 for each in self.actions })
            
            def update_action_frequency():
                length_before = len(tuple(self.action_frequency.keys()))
                self.action_frequency[self.action] += 1
                length_after = len(tuple(self.action_frequency.keys()))
                if length_before < length_after:
                    sort_keys(self.action_frequency)
        # 
        # episodes
        #
        self.all_rewards = 0
        self.episodes = []
        self.per_episode = LazyDict(
            average=LazyDict(
                reward=0,
            ),
        )
        self.timestep = LazyDict(
            index=0,
            observation=None,
            action=None,
            reward=None,
        )
        self.prev_timestep = None
        def new_episode(index):
            self.timestep.observation = self.observation
            self.per_episode.average.reward = self.all_rewards/(len(self.episodes) or 1)
            self.episode = LazyDict(
                index=index,
                timestep=LazyDict(
                    index=0,
                    reward=None,
                ),
                reward=0,
            )
            self.episodes.append(self.episode)
        
        # reward
        def new_reward():
            self.all_rewards += self.reward
            self.episode.reward += self.reward
            
            self.episode.timestep.reward = self.reward
            self.timestep.reward         = self.reward
        
        # timestep
        def post_timestep_start():
            self.timestep.action = self.action
            
        def pre_timestep_end():
            self.prev_timestep = self.timestep
            self.timestep = LazyDict(
                index=self.prev_timestep.index+1,
                observation=self.observation,
                action=None,
                reward=None,
            )
            self.episode.timestep.index += 1
        
        # 
        # prev_observation and action
        # 
        self.prev_observation = None
        def update_prev_observation():
            self.prev_observation          = self.observation
            self.prev_observation_response = self.action
            self.observation               = None
            self.action                    = None
        
        # 
        # when_episode_starts
        # 
        real_when_episode_starts = self.when_episode_starts
        def when_episode_starts(*args, **kwargs):
            new_episode(args[0])
            return real_when_episode_starts(*args, **kwargs)
        self.when_episode_starts = when_episode_starts
        
        # 
        # when_timestep_starts
        # 
        real_when_timestep_starts = self.when_timestep_starts
        def when_timestep_starts(*args, **kwargs):
            output = real_when_timestep_starts(*args, **kwargs)
            post_timestep_start()
            update_action_frequency()
            update_prev_observation()
            return output
        self.when_timestep_starts = when_timestep_starts
        
        # 
        # when_timestep_ends
        # 
        real_when_timestep_ends = self.when_timestep_ends
        def when_timestep_ends(*args, **kwargs):
            pre_timestep_end()
            new_reward()
            
            return real_when_timestep_ends(*args, **kwargs)
        self.when_timestep_ends = when_timestep_ends
            
        return output
    
    return wrapper
