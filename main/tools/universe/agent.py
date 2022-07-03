from tools.universe.timestep import Timestep

class Skeleton:
    # TODO: add a "dont_show_help" and default to listing out all the attributes that enhancements give an agent
    previous_timestep = None
    timestep          = None
    next_timestep     = None
    
    def __init__(self, observation_space, response_space, **config):
        self.observation_space = observation_space
        self.response_space = response_space

    def when_mission_starts(self):
        pass
    def when_episode_starts(self):
        pass
    def when_timestep_starts(self):
        """
        read: self.observation
        write: self.response = something
        """
        pass
    def when_timestep_ends(self):
        """
        read: self.reward
        """
        pass
    def when_episode_ends(self):
        pass
    def when_mission_ends(self):
        pass
    def update_weights(self):
        pass

def enhance_with_single(enhancement_class):
    def wrapper1(init_function):
        def wrapper2(self, *args, **kwargs):
            help(init_function)
            print(f'''init_function = {init_function}''')
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