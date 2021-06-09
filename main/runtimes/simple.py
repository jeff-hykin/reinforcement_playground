def run(env, agent, **config):
    """
    config:
        number_of_episodes: 100
        max_number_of_timesteps: None
    """
    try:
        # 
        # execution loop
        # 
        for each in config["number_of_episodes"]:
            is_over = False
            observation = env.reset()
            agent.on_episode_start(observation)
            raw_reward = 0
            timestep = 0
            while True:
                if type(max_number_of_timesteps) is int and timestep > max_number_of_timesteps:
                    break
                action = agent.decide(observation, raw_reward, is_over)
                raw_reward, observation, is_over, _ = env.step(state)
                if is_over: break
                if agent.wants_to_quit: break
    finally:
        agent.on_clean_up()
        env.close()