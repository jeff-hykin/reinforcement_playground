class MinimalBody:
    observation_space = None
    action_space = None
    wants_to_end_mission = False
    wants_to_end_episode = False
    
    def __init__(self, reality):
        self._reality = reality
    
    def get_reward(self):
        # use a subset of self._reality to decide the reward
        return self._reality.state
        
    def get_observation(self):
        # return a subset of self._reality
        return self._reality.state
    
    def perform_action(self, action):
        # do something to self._reality
        pass
    
    # callbacks that need to be overwritten (use the @ConnectBody)
    when_mission_starts = lambda : pass
    when_episode_starts = lambda episode_index: pass
    when_timestep_happens    = lambda : pass
    when_episode_ends   = lambda episode_index: pass
    when_mission_ends   = lambda : pass

# Class decorator
def Body(Class):
    """
        Example:
            @Body
            class Player1:
                observation_space = None
                action_space = None
                
                def get_reward(self):
                    return world.state.reward # depends on your world setup
    """
    # inherit
    class Body(Class):
        def __init__(self, *args, **kwargs):
            self.wants_to_end_mission = False
            self.wants_to_end_episode = False
            super(Body, self).__init__(*args, **kwargs)
            # connect special methods to the body
            self.when_mission_starts = lambda : pass
            self.when_episode_starts = lambda episode_index: pass
            self.when_timestep_happens    = lambda : pass
            self.when_episode_ends   = lambda episode_index: pass
            self.when_mission_ends   = lambda : pass
            
    return Body


# Class decorator
def ConnectBody(Class):
    """
        Example:
            @ConnectBody
            class Foo:
                def __init__(self, body, config_arg1="hi"):
                    print('config_arg1 = ', config_arg1)
                
                @ConnectBody.when_mission_starts
                def do_stuff(self):
                    print('mission is starting')
                
                @ConnectBody.when_episode_starts
                def do_stuff(self, episode_index):
                    print('episode '+str(episode_index)+' is starting')
    """
    # inherit
    class ConnectedBrain(Class):
        def __init__(self, *args, **kwargs):
            super(ConnectedBrain, self).__init__(*args, **kwargs)
            # connect special methods to the body
            body = kwargs.get("body", None)
            if body is None:
                raise Exception('The class '+str(Class)+' was created with @ConnectBody decorator.\nHowever, the body argument wasn\'t given as an argument (keyword argument) when this object was being created: '+str(self))
            for attribute_name in dir(self):
                each_attribute = getattr(self, attribute_name)
                if callable(each_attribute) and hasattr(each_attribute, '__dict__') and each_attribute.__dict__.get(ConnectBody, False):
                    callback_name = each_attribute.__dict__.get(ConnectBody, False)
                    wrapped_callback = lambda *args, **kwargs: each_attribute(self, *args, **kwargs)
                    # attach the method to the body object
                    setattr(body, callback_name, wrapped_callback)
    return ConnectedBrain

def _when_mission_starts(method):
    method.__dict__ = {}
    method.__dict__[ConnectBody] = "when_mission_starts"
    return method
ConnectBody.when_mission_starts = _when_mission_starts

def _when_episode_starts(method):
    method.__dict__ = {}
    method.__dict__[ConnectBody] = "when_episode_starts"
    return method
ConnectBody.when_episode_starts = _when_episode_starts

def _when_timestep_happens(method):
    method.__dict__ = {}
    method.__dict__[ConnectBody] = "when_timestep_happens"
    return method
ConnectBody.when_timestep_happens = _when_timestep_happens

def _when_episode_ends(method):
    method.__dict__ = {}
    method.__dict__[ConnectBody] = "when_episode_ends"
    return method
ConnectBody.when_episode_ends = _when_episode_ends

def _when_mission_ends(method):
    method.__dict__ = {}
    method.__dict__[ConnectBody] = "when_mission_ends"
    return method
ConnectBody.when_mission_ends = _when_mission_ends


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
            self.before_episode_starts()
        for each_body in self.bodies:
            each_body.when_episode_starts(episode_index)
        if hasattr(self, "after_episode_starts"):
            self.after_episode_starts()
        
    def when_timestep_happens(self):
        if hasattr(self, "before_timestep_happens"):
            self.before_timestep_happens()
        for each_body in self.bodies:
            each_body.when_timestep_happens()
        if hasattr(self, "after_timestep_happens"):
            self.after_timestep_happens()
    
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
