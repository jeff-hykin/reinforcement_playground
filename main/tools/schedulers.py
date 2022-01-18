class AgentLearningRateScheduler:
    """
    Examples:
        base_rate = 0.00001
        additional = 0.001
        total_timesteps = 1000000
        Agent(
            learning_rate=lambda timestep_index, episode_index:  base_rate + additional * (timestep_index/total_timesteps)
        )
    """
    def __init__(self, *, value_function, optimizers):
        self.optimizers = optimizers
        self.timestep_index = -1
        self.episode_index = -1
        # allow the "function" to be a constant
        if not callable(value_function):
            # value_functiontion that returns a constant
            self.value_function = lambda *args: float(value_function)
        else:
            self.value_function = value_function
        # initilize the weights
        self.when_weight_update_starts()
    
    def when_episode_starts(self, episode_index):
        self.episode_index += 1

    def when_timestep_starts(self, timestep_index):
        self.timestep_index += 1
    
    def when_weight_update_starts(self):
        learning_rate = self.current_value
        for each_optimizer in self.optimizers:
            for param_group in each_optimizer.param_groups:
                param_group["lr"] = learning_rate
    
    @property
    def current_value(self):
        return self.value_function(float(self.timestep_index), float(self.episode_index))

class BasicLearningRateScheduler:
    """
    Examples:
        class Model:
            def __init__(self, ):
                self.learning_rate_scheduler = BasicLearningRateScheduler(
                    value_function=self.learning_rate,
                    optimizers=[ self.optimizer ],
                )
            
            def update_weights()
                loss.backward()
                self.learning_rate_scheduler.when_weight_update_starts()
        
        # then use like:
        base_rate = 0.00001
        additional = 0.001
        total_timesteps = 1000000
        Model(
            learning_rate=lambda timestep_index: base_rate + additional * (timestep_index/total_timesteps)
        )
    """
    def __init__(self, *, value_function, optimizers):
        self.optimizers = optimizers
        self.timestep_index = -1
        # allow the "function" to be a constant
        if not callable(value_function):
            # value_functiontion that returns a constant
            self.value_function = lambda *args: float(value_function)
        else:
            self.value_function = value_function
        # initilize the weights
        self.when_weight_update_starts()
    
    def when_weight_update_starts(self):
        self.timestep_index += 1
        learning_rate = self.current_value
        for each_optimizer in self.optimizers:
            for param_group in each_optimizer.param_groups:
                param_group["lr"] = learning_rate
    
    @property
    def current_value(self):
        return self.value_function(float(self.timestep_index))