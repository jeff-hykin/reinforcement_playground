from dataclasses import dataclass, field
@dataclass
class Timestep:
    index       : int   = None
    observation : None  = None
    action      : None  = None
    reward      : float = None
    done        : bool  = None
        
class TimestepSeries:
    def __init__(self, ):
        self._key_offset = 0
        self.index = -1
        self.steps = {}
    
    @property
    def prev(self):
        if self.index > 0:
            return self.steps[self.index-1]
        else:
            return Timestep() # all attributes are none
    
    def add(self, state=None, action=None, reward=None, done=False):
        # if timestep, pull all the data out of the timestep
        if isinstance(state, Timestep):
            observation = state.observation
            action      = state.action
            reward      = state.reward
            done        = state.done
            
        self.index += 1
        self.steps[self.index] = Timestep(self.index, observation, action, reward, done)
    
    @property
    def observations(self):
        return [ each.observation for each in self.steps.values() ]
    
    @property
    def actions(self):
        return [ each.action for each in self.steps.values() ]
    
    @property
    def rewards(self):
        return [ each.reward for each in self.steps.values() ]
    
    @property
    def dones(self):
        return [ each.reward for each in self.steps.values() ]
    
    def items(self):
        """
        for index, state, action, reward, next_state in time_series.items():
            pass
        """
        return ((each.index, each.observation, each.action, each.reward, each.done) for each in self.steps.values())
    
    def __len__(self):
        return len(self.steps)
        
    def __getitem__(self, key):
        if isinstance(key, float):
            key = int(key)
            
        if isinstance(key, int):
            if key < 0:
                key = self.index + key
                if key < 0:
                    return Timestep() # too far back
            # generate steps as needed
            while key > self.index:
                self.add()
            return self.steps[key]
        else:
            new_steps = { 
                each_key: Timestep(*self.steps[each_key])
                    for each_key in range(key.start, key.stop, key.step) 
            }
            time_slice = TimestepSeries()
            time_slice.index = max(new_steps.keys())
            time_slice.steps = new_steps
            return time_slice
    
    def __repr__(self):
        string = "TimestepSeries(\n"
        for index, observation, action, reward, done in self.items():
            string += f"    {index}, {observation}, {action}, {reward}, {done}\n"
        string += ")"
        return string
