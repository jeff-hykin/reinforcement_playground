from super_map import LazyDict
from tools.basics import sort_keys, randomly_pick_from
from tools.universe.agent import Enhancement
from tools.universe.timestep import TimestepSeries, Timestep

class EpisodeEnhancement(Enhancement):
    """
        creates:
            self.episode
            self.episode.index
            self.episode.timestep
            self.episode.reward
            self.episodes[0]
    """
    
    def when_mission_starts(self, original):
        self.episode = LazyDict(
            index=-1,
            timestep=LazyDict(
                index=-1,
                reward=None,
            ),
        )
        self.episodes = []
        self.per_episode = LazyDict(
            average=LazyDict(
                reward=0,
            ),
        )
        
        original()
        
    def when_episode_starts(self, original):
        self.episode = LazyDict(
            index=self.episode.index+1,
            timestep=LazyDict(
                index=-1,
                reward=None,
            ),
            reward=0,
        )
        self.episodes.append(self.episode)
        original()
        
    
    def when_timestep_starts(self, original):
        # get a response
        original()
    
    def when_timestep_ends(self, original):
        # 
        # rewards
        # 
        self.episode.timestep.reward = self.timestep.reward
        self.episode.reward += self.timestep.reward
        
        original()

class TimelineEnhancement(Enhancement):
    """
        needs:
            EpisodeEnhancement
        creates:
            self.timeline
    """
    
    def when_episode_starts(self, original, ):
        self.timeline = TimestepSeries()
        original()
    
    def when_timestep_starts(self, original, ):
        self.timeline[self.episode.timestep.index] = self.timestep
        original()
    
    def when_timestep_ends(self, original, ):
        original()

class LoggerEnhancement(Enhancement):
    """
        needs:
            EpisodeEnhancement
        creates:
            self.per_episode
            self.per_episode.average
            self.per_episode.average.reward
            self.response_frequency    (if self.responses)
    """
    
    def when_mission_starts(self, original):
        # 
        # response_frequency
        # 
        if hasattr(self, "responses") and type(self.responses) != type(None):
            self.response_frequency = LazyDict({ each:0 for each in self.responses })
        
        # 
        # per_episode
        #
        self.all_rewards = 0
        self.per_episode = LazyDict(
            average=LazyDict(
                reward=0,
            ),
        )
        original()
        
    
    def when_timestep_starts(self, original):
        # get a response
        original()
        
        # 
        # update response_frequency
        # 
        if hasattr(self, "response_frequency"):
            length_before = len(tuple(self.response_frequency.keys()))
            self.response_frequency[self.timestep.response] += 1
            length_after = len(tuple(self.response_frequency.keys()))
            if length_before < length_after:
                sort_keys(self.response_frequency)
    
    def when_timestep_ends(self, original):
        # 
        # rewards
        # 
        self.all_rewards    += self.timestep.reward
        self.per_episode.average.reward = self.all_rewards/len(self.episodes)
        
        original()