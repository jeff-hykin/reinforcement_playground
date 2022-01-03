class ListBuffer:
    """
    self.buffer = ListBuffer(
        "rewards",
        "action_log_probabilies",
        "observation_value_estimates",
        "each_action_entropy",
        "was_last_episode_reward",
    )
    self.buffer.add(
        rewards=reward,
        was_last_episode_reward=False,
        each_action_entropy=self.action_entropy,
        action_log_probabilies=self.action_choice_distribution.log_prob(self.action_with_gradient_tracking),
        observation_value_estimates=self.observation_value_estimate,
    )
    """
    def __init__(self, keys):
        self._keys = keys
        self.reset()
    
    def reset(self):
        for each in self._keys:
            setattr(self, each, [])
    
    def add(self,**elements):
        for each_key, each_value in elements.items():
            getattr(self, each_key).append(each_value)
    
    def flush(self):
        output = []
        for each in self._keys:
            output.append(getattr(self, each))
        self.reset()
        return output
    
    def __len__(self):
        for each in self._keys:
            return len(getattr(self, each))
        
    def __getitem__(self, index):
        return tuple(
            getattr(self, each)[index] for each in self._keys
        )
    
    def __iter__(self):
        for each_index in range(len(self)):
            return self[each_index]
    
