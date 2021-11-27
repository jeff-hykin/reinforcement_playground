# Main imports
from world_builders.atari.main import WorldBuilder
from agent_builders.ppo.main import AgentBuilder
from tools.reinverse import Missions
# logging
from tools.record_keeper import ExperimentCollection


# add logging
with ExperimentCollection("logs/record_keeping/ppo_atari_breakout.ignore").new_experiment() as record_keeper:
    # 
    # create the world
    # 
    atari_world = WorldBuilder(game="breakout")

    # 
    # give the agent a body
    # 
    mr_bond = AgentBuilder(
        body=atari_world.bodies[0],
        record_keeper=record_keeper.sub_record_keeper(model="ppo"),
    )

    # 
    # begin mission
    # 
    Missions.simple(
        atari_world,
        max_number_of_episodes=25000,
        max_number_of_timesteps=10000,
    )