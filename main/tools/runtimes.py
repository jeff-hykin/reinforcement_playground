import math

def traditional_runtime(*, agent, env, max_timestep_index=math.inf, max_episode_index=math.inf):
    """
    for episode_index, timestep_index, agent.observation, agent.reward, agent.episode_is_over in traditional_runtime(agent=agent, env=env):
        pass
    """
    agent.when_mission_starts()
    
    episode_index = -1
    while episode_index < max_episode_index:
        episode_index += 1
        
        agent.observation     = env.reset()
        agent.when_episode_starts(episode_index)
        timestep_index = -1
        agent.episode_is_over = False
        while not agent.episode_is_over:
            timestep_index += 1
            
            agent.when_timestep_starts(timestep_index)
            agent.observation, agent.reward, agent.episode_is_over, info = env.step(agent.action)
            agent.when_timestep_ends(timestep_index)
            yield episode_index, timestep_index, agent.observation, agent.reward, agent.episode_is_over
        
        agent.when_episode_ends(episode_index)
    agent.when_mission_ends()

