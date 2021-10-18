def start(reality, max_number_of_timesteps=None, **config):
    """
    config:
        number_of_episodes: 100
        max_number_of_timesteps: None
    """
    try:
        # 
        # execution loop
        # 
        reality.when_campaign_starts()
        for each_agent in reality.agents:
            each_agent.when_campaign_starts()
        for episode_index in range(config["number_of_episodes"]):
            is_over = False
            observation = reality.reset()
            reality.when_episode_starts(episode_index)
            for each_agent in reality.agents:
                each_agent.when_episode_starts(episode_index)
            raw_reward = 0
            timestep = 0
            while True:
                if type(max_number_of_timesteps) is int and timestep > max_number_of_timesteps:
                    break
                reality.when_timestep_happens()
                if reality.wants_to_end_episode: break
                if agent.wants_to_end_episode: break
            
    finally:
        reality.when_campaign_ends()
        for each_agent in reality.agents:
            each_agent.when_campaign_ends()