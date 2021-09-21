def run(env, agent, max_number_of_timesteps=None, **config):
    """
    config:
        number_of_episodes: 100
        max_number_of_timesteps: None
    """
    try:
        # 
        # execution loop
        # 
        for episode_index in range(config["number_of_episodes"]):
            is_over = False
            observation = env.reset()
            agent.on_episode_start(observation, episode_index)
            raw_reward = 0
            timestep = 0
            while True:
                if type(max_number_of_timesteps) is int and timestep > max_number_of_timesteps:
                    break
                action = agent.decide(observation, raw_reward, is_over)
                observation, raw_reward, is_over, _ = env.step(action)
                if is_over: break
                if agent.wants_to_quit: break
    finally:
        agent.on_clean_up()
        env.close()