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
        """
        read: self.observation
        write: self.action = something
        """
        pass
    def when_timestep_ends(self, timestep_index):
        """
        read: self.reward
        """
        pass
    def when_episode_ends(self, episode_index):
        pass
    def when_mission_ends(self, mission_index=0):
        pass
    def update_weights(self):
        pass

def enhance_with_single(enhancement_class):
    def wrapper1(init_function):
        def wrapper2(self, *args, **kwargs):
            output = init_function(self, *args, **kwargs)
            
            real_mission_starts = self.when_mission_starts
            def when_mission_starts(*args, **kwargs):
                return enhancement_class.when_mission_starts(self, real_mission_starts, *args, **kwargs)
            self.when_mission_starts = when_mission_starts
            
            real_episode_starts = self.when_episode_starts
            def when_episode_starts(*args, **kwargs):
                return enhancement_class.when_episode_starts(self, real_episode_starts, *args, **kwargs)
            self.when_episode_starts = when_episode_starts
            
            real_timestep_starts = self.when_timestep_starts
            def when_timestep_starts(*args, **kwargs):
                return enhancement_class.when_timestep_starts(self, real_timestep_starts, *args, **kwargs)
            self.when_timestep_starts = when_timestep_starts
            
            real_timestep_ends = self.when_timestep_ends
            def when_timestep_ends(*args, **kwargs):
                return enhancement_class.when_timestep_ends(self, real_timestep_ends, *args, **kwargs)
            self.when_timestep_ends = when_timestep_ends
            
            real_episode_ends = self.when_episode_ends
            def when_episode_ends(*args, **kwargs):
                return enhancement_class.when_episode_ends(self, real_episode_ends, *args, **kwargs)
            self.when_episode_ends = when_episode_ends
            
            real_mission_ends = self.when_mission_ends
            def when_mission_ends(*args, **kwargs):
                return enhancement_class.when_mission_ends(self, real_mission_ends, *args, **kwargs)
            self.when_mission_ends = when_mission_ends
            
            return output
            
        
        return wrapper2
    return wrapper1

def enhance_with(*enhancements):
    def wrapper(function_getting_wrapped):
        for each_enhancement in enhancements:
            function_getting_wrapped = enhance_with_single(each_enhancement)(function_getting_wrapped)
        return function_getting_wrapped
    return wrapper
        

class Enhancement:
    def when_mission_starts(self, normal_behavior, mission_index=0):
        return normal_behavior(mission_index)
    def when_episode_starts(self, normal_behavior, episode_index):
        return normal_behavior(episode_index)
    def when_timestep_starts(self, normal_behavior, timestep_index):
        return normal_behavior(timestep_index)
    def when_timestep_ends(self, normal_behavior, timestep_index):
        return normal_behavior(timestep_index)
    def when_episode_ends(self, normal_behavior, episode_index):
        return normal_behavior(episode_index)
    def when_mission_ends(self, normal_behavior, mission_index=0):
        return normal_behavior(mission_index)

from super_map import LazyDict
from tools.basics import sort_keys, randomly_pick_from
class AgentBasics(Enhancement):
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
    
    def when_mission_starts(self, normal_behavior, mission_index=0):
        print(f'''self, normal_behavior, mission_index=0 = {(self, normal_behavior, mission_index)}''')
        # 
        # prev_observation
        # 
        self.prev_observation = None
        self.prev_observation_response = None
        
        # 
        # action_frequency
        # 
        if hasattr(self, "actions") and type(self.actions) != type(None):
            self.action_frequency = LazyDict({ each:0 for each in self.actions })
        
        # 
        # episodes
        #
        self.all_rewards = 0
        self.prev_timestep = None
        self.timestep = LazyDict(
            index=0,
            observation=None,
            action=None,
            reward=None,
        )
        self.episodes = []
        self.per_episode = LazyDict(
            average=LazyDict(
                reward=0,
            ),
        )
        
        normal_behavior(mission_index)
        
    def when_episode_starts(self, normal_behavior, episode_index):
        self.timestep.observation = self.observation
        self.per_episode.average.reward = self.all_rewards/(len(self.episodes) or 1)
        self.episode = LazyDict(
            index=episode_index,
            timestep=LazyDict(
                index=0,
                reward=None,
            ),
            reward=0,
        )
        self.episodes.append(self.episode)
        
        normal_behavior(episode_index)
        
    
    def when_timestep_starts(self, normal_behavior, timestep_index):
        
        normal_behavior(timestep_index)
        
        # 
        # update action
        # 
        self.timestep.action = self.action
        
        # 
        # update action_frequency
        # 
        if hasattr(self, "action_frequency"):
            length_before = len(tuple(self.action_frequency.keys()))
            self.action_frequency[self.action] += 1
            length_after = len(tuple(self.action_frequency.keys()))
            if length_before < length_after:
                sort_keys(self.action_frequency)
        
        # 
        # set prev_observation
        # 
        self.prev_observation          = self.observation
        self.prev_observation_response = self.action
        self.observation               = None
        self.action                    = None
    
    def when_timestep_ends(self, normal_behavior, timestep_index):
        # 
        # rewards
        # 
        self.all_rewards    += self.reward
        self.episode.reward += self.reward
        
        self.episode.timestep.reward = self.reward
        self.timestep.reward         = self.reward
        
        # 
        # timestep update
        # 
        self.prev_timestep = self.timestep
        self.timestep = LazyDict(
            index=self.prev_timestep.index+1,
            observation=self.observation,
            action=None,
            reward=None,
        )
        self.episode.timestep.index += 1
        
        normal_behavior(timestep_index)
    