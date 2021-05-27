from tools.defaults import *
config = merge(config, ez_yaml.to_object(file_path=join(dirname(__file__),"config.yaml")))

# 
# entrypoint
# 
if __name__ == '__main__':
    # merge in the local config
    
    # 
    # vae
    # 
    from tools.jirl_vae import Vae
    vae = Vae(**config["vae"])
    
    # 
    # env
    # 
    from environments.unity_car_track.environment import Environment
    env = Environment(vae=vae, **config["environment"])
    
    #
    # agent
    #
    from agents.sac import Agent
    agent = Agent(**config["agent"])
    
    # 
    # execution loop
    # 
    for each in config["iterations"]:
        state = env.reset()
        raw_reward = 0
        while True:
            action = agent.decide(state, raw_reward)
            raw_reward, state, is_over, _ = env.step(state)
            if is_over: break
            if agent.wants_to_quit: break
    
    agent.clean_up() # saves checkpoints, logs
    vae.clean_up()   # saves parameters
    env.clean_up()   # saves timing data