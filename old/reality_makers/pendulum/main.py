import gym
from gym import spaces
from tools.reality_maker import MinimalReality, MinimalBody

class RealityMaker(MinimalReality):
    
    def __init__(self, *, agents, version=0, **config):
        super(RealityMaker, self).__init__(agents=agents)
        self._env = gym.make(f'Pendulum-v{version}')
        self.debugging_info = None
        self.wants_to_end_episode = False
    
    def when_episode_starts(self):
        # the state of the pendulum, and the state of the reward
        self.state = (self._env.reset(), None)
        self.wants_to_end_episode = False
    
    def when_timestep_happens(self):
        for agent in self.agents:
            agent.when_timestep_happens()
        # update the reality
        self.state[0], self.state[1], self.wants_to_end_episode, self.debugging_info = self._env.step(agent.body.action)
    
    def when_campaign_ends(self):
        self._env.close()
    
    class RegularBody(MinimalBody):
        def __init__(self, options):
            super(RealityMaker.RegularBody, self).__init__()
            self.action = None
        
        @property
        def action_space(self):
            return self._reality._env.action_space
        
        @property
        def observation_space(self):
            return self._reality._env.observation_space
            
        def get_observation(self):
            # return a subset of self._reality
            return self._reality.state[0]
            
        def get_reward(self):
            return self._reality.state[1]
            
        def perform_action(self, action):
            self.action = action