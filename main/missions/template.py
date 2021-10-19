from world_builders.atari.main import WorldBuilder
from brain_builders.ppo.main import BrainBuilder

def main():
    # 
    # create the world
    # 
    enduro_world = WorldBuilder(game="enduro")
    # 
    # connect the brain to the body
    # 
    mr_bond = BrainBuilder(
        body=enduro_world.bodies[0],
    )
    # 
    # begin mission
    # 
    begin_mission(enduro_world, max_number_of_episodes=100, max_number_of_timesteps=1000)


# helper / runtime
def begin_mission(world, max_number_of_episodes=float("inf"), max_number_of_timesteps=float("inf")):
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

main()