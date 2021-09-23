class MinimalBody:
    observation_space = None
    action_space = None
    
    def __init__(self):
        self._reality = None
    
    def get_reward(self):
        # use a subset of self._reality to decide the reward
        return self._reality.state
        
    def get_observation(self):
        # return a subset of self._reality
        return self._reality.state
    
    def perform_action(self, action):
        # do something to self._reality
        pass

class MinimalAgent:
    def __init__(self, body_type):
        # required poperties
        self.body_type = None
        self.body_options = None
        self.body = None
        self.wants_to_quit = False
    
    def when_body_is_ready(self): pass
    def when_campaign_starts(self): pass
    def when_episode_starts(self, episode_index): pass
    def when_time_passes(self): pass
    def when_episode_ends(self, episode_index): pass
    def when_campaign_ends(self): pass
        
class MinimalReality:
    
    def __init__(self, agents):
        self.agents = agents
        # setup the body for all the agents
        for each_agent in self.agents:
            # make sure the agent has a body_type
            possible_body_class = getattr(each_agent, "body_type", None)
            if not possible_body_class or not issubclass(possible_body_class, MinimalBody):
                raise Exception(f'Error, when registering an agent, the agent.body_type type was {possible_body_class} which is not one of available options.\nUse this_reality.body_types to get the types, which are: {self.body_types}')
    
    @property 
    def body_types(self):
        # collect all the body types
        body_types = []
        for each_attribute in dir(self):
            potential_class = getattr(self, each_attribute, None)
            if issubclass(potential_class, MinimalBody):
                body_types.append(potential_class)
        return body_types
    
    def when_campaign_starts(self):
        """
            can be used as a once-per-experiment event
            (ex: establishing external process)
        """
        for each_agent in self.agents:
            # create bodies and connect them to reality
            each_agent.body = each_agent.body_type(each_agent.body_options)
            each_agent.body._reality = self
            # then tell the agent their body is ready
            each_agent.when_body_is_ready()
            
    def when_episode_starts(self, episode_index):
        """
            use this to set/reset reality
        """
        pass
        
    def when_time_passes(self):
        for agent in self.agents:
            agent.when_time_passes()
    
    def when_episode_ends(self, episode_index):
        """
            similar to when_episode_starts, this can be used to set/reset reality
        """
        pass
    
    def when_campaign_ends(self):
        """
            this is a cleanup step
        """
        pass
