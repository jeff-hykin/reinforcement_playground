from world_builders.atari.main import WorldBuilder
from agent_builders.ppo.main import AgentBuilder
from tools.reinverse import Missions

# 
# create the world
# 
enduro_world = WorldBuilder(game="enduro")

# 
# give the agent a body
# 
mr_bond = AgentBuilder(
    body=enduro_world.bodies[0],
)

# 
# begin mission
# 
Missions.simple(
    enduro_world,
    max_number_of_episodes=10000,
    max_number_of_timesteps=10000,
)