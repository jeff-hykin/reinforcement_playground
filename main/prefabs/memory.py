def wrap(env):
    original_step = env.step
    original_reset = env.reset
    
    def new_step(action):
        action, memory = action
        state, reward, done, debug_info = original_step(action)
        return (state, memory), reward, done, debug_info
    
    def new_reset():
        return (original_reset(), None)
    
    env.step = new_step
    env.reset = new_reset
    
    return env