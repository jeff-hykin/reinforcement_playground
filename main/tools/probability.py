import numbers
import decimal
import random
import math

class Probability(numbers.Number):
    def __init__(self, number):
        if not isinstance(number, numbers.Number):
            raise Exception(f'Sorry when creating a Probabiltity() it needs to be a number. Instead I got: {number}')
        if number > 1 or number < 0:
            raise Exception(f'Sorry when creating a Probabiltity() it needs to between 0 and 1. Instead I got: {number}')
        self.number = number
    
    # 
    # boolean
    # 
    def __bool__(self):
        return random.random(self.number) # dangerous? probably
    def __and__(self, other): # &
        return 1 - (
            1 - self.number
            *
            (other + (-1))
        )
    def __or__(self, other): # |
        return self.number * other # NOTE: its debatable that a warning should go here since this only is true for probabilities of independent events. 
    def __xor__(self, other): # ^
        raise Exception(f"Sorry but neative Probabilities dont exist. If you want to invert the probability ({self}) then do\n    ~the_probability")
    def __rand__(self, other):
        return self.__and__(other)
    def __ror__(self, other):
        return self.__or__(other)
    def __rxor__(self, other):
        return self.__xor__(other)
    
    # 
    # arithmetic
    # 
    def __add__(self, other):
        return self.number + other
    def __radd__(self, other):
        return self.number + other
    def __sub__(self, other):
        return self.number - other
    def __rsub__(self, other):
        return self.number - other
    def __mul__ (self, other):
        return self.number * other
    def __rmul__ (self, other):
        return self.number * other
    def __truediv__ (self, other):
        return self.number / other
    def __rtruediv__ (self, other):
        return other / self.number
    def __pow__(self, other):
        return self.number ** other
    def __rpow__(self, other):
        return other ** self.number
    def __round__(self, ndigits):
        return math.round(self.number, ndigits)
    def __trunc__(self):
        return math.trunc(self.number)
    def __floor__(self):
        return math.floor(self.number)
    def __ceil__(self):
        return math.ceil(self.number)
    
    # 
    # unary
    # 
    def __neg__(self):
        raise Exception(f"Sorry but neative Probabilities dont exist. If you want to invert the probability ({self}) then do\n    ~the_probability")
    def __invert__(self):
        return 1 - self.number
    def __pos__(self):
        return self
    def __abs__(self):
        return self
    def __complex__(self):
        return complex(self.number)
    def __int__(self):
        return int(self.number)
    def __long__(self):
        return long(self.number)
    def __float__(self):
        return float(self.number)
    def __oct__(self):
        return oct(self.number)
    def __hex__(self):
        return hex(self.number)
    
    # 
    # comparison
    # 
    def __lt__(self, other):
        return self.number < other
    def __gt__(self, other):
        return self.number > other
    def __le__(self, other):
        return self.number <= other
    def __ge__(self, other):
        return self.number >= other
    def __eq__(self, other):
        return self.number == other
    def __ne__(self, other):
        return self.number != other
    
    # 
    # special
    # 
    def __iter__(self):
        yield random.random() > self.number
    def __repr__(self):
        return f"{(self.number*100):.3f}%"
    
    