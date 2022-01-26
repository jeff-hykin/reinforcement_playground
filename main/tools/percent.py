import numbers
import decimal
import random
import math

class Percent(numbers.Number):
    def __init__(self, number):
        if not isinstance(number, numbers.Number):
            raise Exception(f'Sorry when creating a Percent() it needs to be a number. Instead I got: {number}')
        self.proportion = number/100
    
    # 
    # boolean
    # 
    def __bool__(self):
        return bool(self.proportion)
    
    # 
    # arithmetic
    # 
    def __add__(self, other):
        return self.proportion + other
    def __radd__(self, other):
        return self.proportion + other
    def __sub__(self, other):
        return self.proportion - other
    def __rsub__(self, other):
        return self.proportion - other
    def __mul__ (self, other):
        return self.proportion * other
    def __rmul__ (self, other):
        return self.proportion * other
    def __truediv__ (self, other):
        return self.proportion / other
    def __rtruediv__ (self, other):
        return other / self.proportion
    def __pow__(self, other):
        return self.proportion ** other
    def __rpow__(self, other):
        return other ** self.proportion
    def __round__(self, ndigits):
        return Percent(math.round(self.proportion * 100, ndigits))
    def __trunc__(self):
        return Percent(math.trunc(self.proportion * 100))
    def __floor__(self):
        return Percent(math.floor(self.proportion * 100))
    def __ceil__(self):
        return Percent(math.ceil(self.proportion * 100))
    
    # 
    # unary
    # 
    def __neg__(self):
        raise Percent(-self.proportion * 100)
    def __invert__(self):
        return Percent((1 - self.proportion)*100)
    def __pos__(self):
        return Percent(abs(self.proportion))
    def __abs__(self):
        return Percent(abs(self.proportion))
    def __complex__(self):
        return complex(self.proportion) * 100
    def __int__(self):
        return int(self.proportion) * 100
    def __long__(self):
        return long(self.proportion) * 100
    def __float__(self):
        return float(self.proportion) * 100
    def __oct__(self):
        return oct(self.proportion) * 100
    def __hex__(self):
        return hex(self.proportion) * 100
    
    # 
    # comparison
    # 
    def __lt__(self, other):
        return self.proportion < other
    def __gt__(self, other):
        return self.proportion > other
    def __le__(self, other):
        return self.proportion <= other
    def __ge__(self, other):
        return self.proportion >= other
    def __eq__(self, other):
        return self.proportion == other
    def __ne__(self, other):
        return self.proportion != other
    
    # 
    # special
    # 
    def __repr__(self):
        return f"{(self.proportion*100):.3f}%"
    def __format__(self, format):
        return f'Value({(self.proportion*100):{format}})'
    
    