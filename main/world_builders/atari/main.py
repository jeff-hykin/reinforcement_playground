from super_map import Map, LazyDict

# local
from tools.reinverse import WorldBuilder, MinimalWorld, BodyBuilder, MinimalBody
from world_builders.atari.environment import Environment

@WorldBuilder
class WorldBuilder(MinimalWorld):
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
    def __init__(world, *args, **kwargs):
        world.game = Environment(**kwargs)
        world.state = None
        world.debugging_info = None
        
        # define a body and create it
        @BodyBuilder
        class Player(MinimalBody):
            observation_space = world.game.observation_space
            action_space      = world.game.action_space
            
            def get_observation(self):
                return world.state.image
            
            def get_reward(self):
                return world.state.score
            
            def perform_action(self, action):
                self.action = action
        
        # this name is specific!
        world.bodies = [
            Player(),
        ]
    
    def before_episode_starts(self, episode_index):
        self.state = LazyDict(
            image=self.game.reset(),
            score=0,
        )
    
    def after_timestep_happens(self, timestep_index):
        player_1 = self.bodies[0]
        # act randomly when no action given (for DEBUGGING, this is not good general practice)
        if player_1.action is None:
            player_1.action = player_1.action_space.sample()
        # update the state, and episode status
        self.state.image, self.state.score, self.wants_to_end_episode, self.debugging_info = self.game.step(player_1.action)
    
    def when_mission_ends(self):
        self.game.close()