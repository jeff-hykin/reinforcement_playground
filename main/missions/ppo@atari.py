from world_builders.atari.main import WorldBuilder
from agent_builders.ppo.main import AgentBuilder
from tools.reinverse import Missions

# 
# create the world
# 
atari_world = WorldBuilder(game="breakout")

# 
# give the agent a body
# 
mr_bond = AgentBuilder(
    body=atari_world.bodies[0],
)

# 
# begin mission
# 
Missions.simple(
    atari_world,
    max_number_of_episodes=25000,
    max_number_of_timesteps=10000,
)