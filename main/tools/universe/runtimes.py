import itertools
import math
from copy import deepcopy
from tools.universe.timestep import Timestep


def basic(*, agent, env, max_timestep_index=math.inf, max_episode_index=math.inf):
    """
    for episode_index, timestep_index, agent.timestep.observation, agent.timestep.reward, agent.timestep.is_last_step in traditional_runtime(agent=agent, env=env):
        pass
    """
    agent.when_mission_starts()
    
    for episode_index in itertools.count(0): # starting at 0
        
        agent.previous_timestep = Timestep(
            index=-2,
        )
        agent.timestep = Timestep(
            index=-1,
        )
        agent.next_timestep = Timestep(
            index=0,
            observation=deepcopy(env.reset()),
            is_last_step=False,
        )
        agent.when_episode_starts()
        while not agent.timestep.is_last_step:
            
            agent.previous_timestep = agent.timestep
            agent.timestep          = agent.next_timestep
            agent.next_timestep     = Timestep(index=agent.next_timestep.index+1)
            
            agent.when_timestep_starts()
            if type(agent.timestep.reaction) == type(None):
                agent.timestep.reaction = env.action_space.sample()
            observation, reward, is_last_step, agent.timestep.hidden_info = env.step(agent.timestep.reaction)
            agent.next_timestep.observation = deepcopy(observation)
            agent.timestep.reward           = deepcopy(reward)
            agent.timestep.is_last_step     = deepcopy(is_last_step)
            agent.when_timestep_ends()
            
            yield episode_index, agent.timestep
        
        agent.when_episode_ends()
    agent.when_mission_ends()