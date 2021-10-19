class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def equalize(self):
        smallest_size = min(
            len(self.actions),
            len(self.states),
            len(self.logprobs),
            len(self.rewards),
            len(self.is_terminals),
        )
        # truncate to equalize
        self.actions       = self.actions[0:smallest_size]
        self.states        = self.states[0:smallest_size]
        self.logprobs      = self.logprobs[0:smallest_size]
        self.rewards       = self.rewards[0:smallest_size]
        self.is_terminals  = self.is_terminals[0:smallest_size]

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]