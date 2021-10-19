class RolloutBuffer:
    def __init__(self):
        # main data
        self.actions            = []
        self.states             = []
        self.logprobs           = []
        self.rewards            = []
        self.is_terminals       = []
        # helpers
        self._init_actions       = []
        self._init_states        = []
        self._init_logprobs      = []
        self._init_rewards       = []
        self._init_is_terminals  = []
    
    def equalize(self):
        smallest_size = min(
            len(self.actions),
            len(self.states),
            len(self.logprobs),
            len(self.rewards),
            len(self.is_terminals),
        )
        # save what is about to get truncated
        self._init_actions       = self.actions[smallest_size:]
        self._init_states        = self.states[smallest_size:]
        self._init_logprobs      = self.logprobs[smallest_size:]
        self._init_rewards       = self.rewards[smallest_size:]
        self._init_is_terminals  = self.is_terminals[smallest_size:]
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
        # init them with was was chopped off from before
        self.actions       = self._init_actions
        self.states        = self._init_states
        self.logprobs      = self._init_logprobs
        self.rewards       = self._init_rewards
        self.is_terminals  = self._init_is_terminals