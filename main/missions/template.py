from world_builders.atari.main import WorldBuilder
from brain_builders.ppo import BrainBuilder

# create the world
pong_world = WorldBuilder(game="pong")
# connect the brain to the body
mr_bond = BrainBuilder(
    body=pong_world.player_1
)

def start_mission(world, max_number_of_episodes=math.inf, max_number_of_timesteps=math.inf):
    import itertools
    import math
    try:
        world.when_mission_starts()
        for episode_index in itertools.count(0): # starting at 0, count forever
            if index > max_number_of_episodes:
                break
            world.when_episode_starts(episode_index)
            for timestep_index in itertools.count(0): # starting at 0, count forever
                if timestep_index > max_number_of_timesteps:
                    break
                # check for early end
                if world.wants_to_end_episode: break
                for each_body in world.bodies:
                    if each_body.wants_to_end_episode: break
                world.when_time_passes(timestep)
            # check for early end
            if world.wants_to_end_mission: break
            for each_body in world.bodies:
                if each_body.wants_to_end_mission: break
            
    finally:
        world.when_mission_ends()


start_mission(pong_world, max_number_of_episodes=100, max_number_of_timesteps=1000)