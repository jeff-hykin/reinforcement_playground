# Main imports
from world_builders.atari.main import WorldBuilder
from agent_builders.ppo.main import AgentBuilder
from tools.reinverse import Missions
# logging
from tools.record_keeper import ExperimentCollection
from tools.liquid_data import LiquidData
from tools.debug import debug
import silver_spectacle as ss


# add logging
with ExperimentCollection("logs/record_keeping/ppo_atari_breakout.ignore").new_experiment() as record_keeper:
    debug.record_keeper = record_keeper
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
        max_number_of_episodes=100,
        max_number_of_timesteps=10000,
    )
    
    # 
    # Display results
    # 
    by_update_records = tuple(each for each in record_keeper if each["by_update"])
    y_values = tuple( each["reward"]       for each in by_update_records )
    x_values = tuple( each["update_index"] for each in by_update_records )
    ss.DisplayCard("quickLine", list(zip(x_values, y_values)))