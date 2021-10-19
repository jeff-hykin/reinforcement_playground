from simple_namespace import namespace

# 
# 
# Body
#
#  

# Class decorator
def BodyBuilder(Class):
    """
        Example:
            @BodyBuilder
            class Player1(MinimalBody):
                observation_space = None
                action_space = None
                
                def get_reward(self):
                    return world.state.reward # depends on your world setup
    """
    # inherit
    class BodyBuilder(Class):
        def __init__(self, *args, **kwargs):
            self.wants_to_end_mission = False
            self.wants_to_end_episode = False
            self.action = None
            # the Brain will overwrite all of these lambdas
            self.when_mission_starts   = lambda               : None
            self.when_episode_starts   = lambda episode_index : None
            self.when_timestep_happens = lambda timestep_index: None
            self.when_episode_ends     = lambda episode_index : None
            self.when_mission_ends     = lambda               : None
            super(BodyBuilder, self).__init__(*args, **kwargs)
            
    return BodyBuilder

class MinimalBody:
    observation_space = None
    action_space = None
    # callbacks that need to be overwritten (use the @ConnectBody)
    # (these are redundantly placed here to help python-autocomplete tools)
    when_mission_starts   = lambda               : None
    when_episode_starts   = lambda episode_index : None
    when_timestep_happens = lambda timestep_index: None
    when_episode_ends     = lambda episode_index : None
    when_mission_ends     = lambda               : None
    
    def get_reward(self):
        # use a subset of self._reality to decide the reward
        return self._reality.state
        
    def get_observation(self):
        # return a subset of self._reality
        return self._reality.state
    
    def perform_action(self, action):
        # do something to self._reality
        pass

# 
# 
# Agent
# 
# 

# helper
def add_special_decorator(main_object, method_name):
    def wrapper(method):
        method.__dict__ = {}
        method.__dict__[main_object] = method_name
        return method
    setattr(main_object, method_name, wrapper) 

# Class decorator
def ConnectBody(Class):
    """
        Example:
            @ConnectBody
            class AgentBuilder:
                def __init__(self, body, config_arg1="hi"):
                    self.body = body
                    self.config_arg1 = self.config_arg1
                
                @ConnectBody.when_mission_starts
                def when_mission_starts(self):
                    print('mission is starting')
                
                @ConnectBody.when_episode_starts
                def when_episode_starts(self, episode_index):
                    print('episode '+str(episode_index)+' is starting')
                
                @ConnectBody.when_timestep_happens
                def when_timestep_happens(self, timestep_index):
                    if timestep_index > 0:
                        reward = self.body.get_reward()
                    
                    observation = self.body.get_observation()
                    action = self.body.action_space.sample()
                    self.body.perform_action(action)
    """
    # inherit
    class ConnectedAgent(Class):
        def __init__(self, *args, **kwargs):
            super(ConnectedAgent, self).__init__(*args, **kwargs)
            # connect special methods to the body
            body = kwargs.get("body", None)
            if body is None:
                raise Exception('The class '+str(Class)+' was created with @ConnectBody decorator.\nHowever, the body argument wasn\'t given as an argument (keyword argument) when this object was being created: '+str(self))
            for attribute_name in dir(self):
                each_attribute = getattr(self, attribute_name)
                if callable(each_attribute) and hasattr(each_attribute, '__dict__') and each_attribute.__dict__.get(ConnectBody, False):
                    callback_name = each_attribute.__dict__.get(ConnectBody, False)
                    def scope_fixer():
                        local_copy_of_attribute = each_attribute
                        return lambda *args, **kwargs: local_copy_of_attribute(*args, **kwargs)
                    # attach the method to the body object
                    setattr(body, callback_name, scope_fixer())
    return ConnectedAgent

add_special_decorator(ConnectBody, "when_mission_starts")
add_special_decorator(ConnectBody, "when_episode_starts")
add_special_decorator(ConnectBody, "when_timestep_happens")
add_special_decorator(ConnectBody, "when_episode_ends")
add_special_decorator(ConnectBody, "when_mission_ends")

# 
# 
# World
#
#  

# Class decorator
def WorldBuilder(Class):
    """
        Example:
            @WorldBuilder
            class WorldBuilder:
                def __init__(self, config_arg1="hi"):
                    self.config_arg1 = self.config_arg1
                
                def when_mission_starts(self):
                    print('mission is starting')
                
                def when_episode_starts(self, episode_index):
                    print('episode '+str(episode_index)+' is starting')
                
                def when_timestep_happens(self, timestep_index):
                    if timestep_index > 0:
                        reward = self.body.get_reward()
                    
                    observation = self.body.get_observation()
                    action = self.body.action_space.sample()
                    self.body.perform_action(action)
    """
    # inherit
    class World(Class):
        def __init__(self, *args, **kwargs):
            self.wants_to_end_episode = False
            self.wants_to_end_mission = False
            self.bodies = []
            super(World, self).__init__(*args, **kwargs)
        
        def when_mission_starts(self, *args, **kwargs):
            super(World, self).when_mission_starts(*args, **kwargs)
            if len(self.bodies) <= 0:
                raise Exception(f"There's a class ({Class}) marked with @WorldBuilder that doesn't create a self.bodies attribute\nWorldBuilder needs:\n    self.bodies (list of body objects, see `BodyBuilder`)\n    self.wants_to_end_episode (true/false)\n    self.wants_to_end_mission (true/false)\nThose are the only special attributes that need to be set")
    return World

class MinimalWorld:
    """
        required attributes:
            self.bodies # needs to be an iterable of body objects (see the MinimalBody class)
            self.wants_to_end_episode # boolean
            self.wants_to_end_mission # boolean
        required methods:
            None
        available methods:
            before_mission_starts()
            when_mission_starts() # this will override existing behavior
            after_mission_starts()
            
            before_episode_starts()
            when_episode_starts() # this will override existing behavior
            after_episode_starts()
            
            before_timestep_happens()
            when_timestep_happens() # this will override existing behavior
            after_timestep_happens()
            
            before_episode_ends()
            when_episode_ends() # this will override existing behavior
            after_episode_ends()
            
            before_mission_ends()
            when_mission_ends() # this will override existing behavior
            after_mission_ends()
    """
    
    def when_mission_starts(self):
        """
            can be used as a once-per-experiment event
            (ex: establishing external process)
        """
        if hasattr(self, "before_mission_starts"):
            self.before_mission_starts()
        for each_body in self.bodies:
            each_body.when_mission_starts()
        if hasattr(self, "after_mission_starts"):
            self.after_mission_starts()
            
    def when_episode_starts(self, episode_index):
        """
            use this to set/reset reality
        """
        if hasattr(self, "before_episode_starts"):
            self.before_episode_starts(episode_index)
        for each_body in self.bodies:
            each_body.when_episode_starts(episode_index)
        if hasattr(self, "after_episode_starts"):
            self.after_episode_starts(episode_index)
        
    def when_timestep_happens(self, timestep_index):
        if hasattr(self, "before_timestep_happens"):
            self.before_timestep_happens(timestep_index)
        for each_body in self.bodies:
            each_body.when_timestep_happens(timestep_index)
        if hasattr(self, "after_timestep_happens"):
            self.after_timestep_happens(timestep_index)
    
    def when_episode_ends(self, episode_index):
        """
            similar to when_episode_starts, this can be used to set/reset reality
        """
        if hasattr(self, "before_episode_ends"):
            self.before_episode_ends()
        for each_body in self.bodies:
            each_body.when_episode_ends()
        if hasattr(self, "after_episode_ends"):
            self.after_episode_ends()
    
    def when_mission_ends(self):
        """
            this is a cleanup step
        """
        if hasattr(self, "before_mission_ends"):
            self.before_mission_ends()
        for each_body in self.bodies:
            each_body.when_mission_ends()
        if hasattr(self, "after_mission_ends"):
            self.after_mission_ends()



# 
# 
# mission helpers
# 
# 
@namespace
def Missions():
    
    def simple(world, max_number_of_episodes=float("inf"), max_number_of_timesteps=float("inf")):
        import itertools
        try:
            world.when_mission_starts()
            # 
            # episodes
            # 
            for episode_index in itertools.count(0): # starting at 0, count forever
                # break conditions
                if episode_index > max_number_of_episodes:
                    break
                if world.wants_to_end_mission:
                    world.wants_to_end_mission = False
                    break
                if any(each_body.wants_to_end_mission for each_body in world.bodies):
                    for each_body in world.bodies: each_body.wants_to_end_mission = False
                    break
                
                world.when_episode_starts(episode_index)
                
                # 
                # timesteps
                # 
                for timestep_index in itertools.count(0): # starting at 0, count forever
                    if timestep_index > max_number_of_timesteps:
                        break
                    if world.wants_to_end_episode:
                        world.wants_to_end_episode = False
                        break
                    if any(each_body.wants_to_end_episode for each_body in world.bodies):
                        for each_body in world.bodies: each_body.wants_to_end_episode = False
                        break
                    
                    world.when_timestep_happens(timestep_index)
                
        finally:
            # 
            # end timestep
            # 
            world.when_mission_ends()
    
    return locals()
