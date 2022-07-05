from dataclasses import dataclass, field
@dataclass
class Timestep:
    index        : int   = None
    observation  : None  = None 
    response     : None  = None 
    reward       : float = None 
    is_last_step : bool  = False
    hidden_info  : None  = None
    
    def __init__(self, timestep=None, *, index=None, observation=None, response=None, reward=None, is_last_step=None, hidden_info=None):
        if timestep:
            for each_attr in dir(timestep):
                # skip magic attributes
                if len(each_attr) > 2 and each_attr[0:2] == '__':
                    continue
                # adopt all other attributes
                setattr(self, each_attr, getattr(timestep, each_attr))
        
        # set any non-None values
        self.index        = index        if not (type(index       ) == type(None)) else self.index
        self.observation  = observation  if not (type(observation ) == type(None)) else self.observation
        self.response     = response     if not (type(response    ) == type(None)) else self.response
        self.reward       = reward       if not (type(reward      ) == type(None)) else self.reward
        self.is_last_step = is_last_step if not (type(is_last_step) == type(None)) else self.is_last_step
        self.hidden_info  = hidden_info  if not (type(hidden_info ) == type(None)) else self.hidden_info

class TimestepSeries:
    def __init__(self, ):
        self.index = -1
        self.steps = {}
    
    @property
    def prev(self):
        if self.index > 0:
            return self.steps[self.index-1]
        else:
            return Timestep() # all attributes are none/false
    
    def add(self, state=None, response=None, reward=None, is_last_step=False):
        # if timestep, pull all the data out of the timestep
        if isinstance(state, Timestep):
            observation  = state.observation
            response     = state.response
            reward       = state.reward
            is_last_step = state.is_last_step
            
        self.index += 1
        self.steps[self.index] = Timestep(index=self.index, observation=observation, response=response, reward=reward, is_last_step=is_last_step)
    
    @property
    def observations(self):
        return [ each.observation for each in self.steps.values() ]
    
    @property
    def responses(self):
        return [ each.response for each in self.steps.values() ]
    
    @property
    def rewards(self):
        return [ each.reward for each in self.steps.values() ]
    
    @property
    def is_last_steps(self):
        return [ each.reward for each in self.steps.values() ]
    
    def items(self):
        """
        for index, state, response, reward, next_state in time_series.items():
            pass
        """
        return ((each.index, each.observation, each.response, each.reward, each.is_last_step) for each in self.steps.values())
    
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
                each_key: Timestep(self.steps[each_key])
                    for each_key in range(key.start, key.stop, key.step) 
            }
            time_slice = TimestepSeries()
            time_slice.index = max(new_steps.keys())
            time_slice.steps = new_steps
            return time_slice
    
    def __repr__(self):
        string = "TimestepSeries(\n"
        for index, observation, response, reward, is_last_step in self.items():
            string += f"    {index}, {observation}, {response}, {reward}, {is_last_step}\n"
        string += ")"
        return string
