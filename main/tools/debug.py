# 
# debug var
# 
from super_map import LazyDict
# simple but effective 
debug = LazyDict()
# (import this value into other modules, and set attributes to watch values)

# 
# icecream
# 
from icecream import ic

# 
# actually_pretty_print
# 
# from https://stackoverflow.com/a/26209900/4367134
class Formatter(object):
    def __init__(self):
        self.types = {}
        self.htchar = '    '
        self.lfchar = '\n'
        self.indent = 0
        self.set_formater(object, self.__class__.format_object)
        self.set_formater(dict, self.__class__.format_dict)
        self.set_formater(list, self.__class__.format_list)
        self.set_formater(tuple, self.__class__.format_tuple)
    
    def set_formater(self, obj, callback):
        self.types[obj] = callback
    
    def __call__(self, value, **args):
        for key in args:
            setattr(self, key, args[key])
        formater = self.types[type(value) if type(value) in self.types else object]
        return formater(self, value, self.indent)
    
    def format_object(self, value, indent):
        return repr(value)
    
    def format_dict(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + repr(key) + ': ' +
            (self.types[type(value[key]) if type(value[key]) in self.types else object])(self, value[key], indent + 1)
            for key in value
        ]
        return '{%s}' % (','.join(items) + self.lfchar + self.htchar * indent)
    
    def format_list(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + (self.types[type(item) if type(item) in self.types else object])(self, item, indent + 1)
            for item in value
        ]
        return '[%s]' % (','.join(items) + self.lfchar + self.htchar * indent)
    
    def format_tuple(self, value, indent):
        items = [
            self.lfchar + self.htchar * (indent + 1) + (self.types[type(item) if type(item) in self.types else object])(self, item, indent + 1)
            for item in value
        ]
        return '(%s)' % (','.join(items) + self.lfchar + self.htchar * indent)

pretty = Formatter()

# 
# yapf pretty_print
# 
# def pretty_print(each):
#     import builtins
#     builtins.print('each = ', each)
#     the_pformat = pformat(each)
#     builtins.print('the_pformat = ', the_pformat)
#     formatted_code, _ = FormatCode(the_pformat)
#     builtins.print('formatted_code = ', formatted_code)
#     return formatted_code

ic.configureOutput(argToStringFunction=pretty)

# 
# indentable print
# 
import sys
original_print = print
def print(*args, **kwargs):
    from io import StringIO
    string_stream = StringIO()
    original_print(*args, **{**kwargs, "file": string_stream},)
    output_str = string_stream.getvalue()
    string_stream.close()
    indent = (" "*print.indent)
    output_str = indent + output_str.replace("\n", "\n"+indent)
    return original_print(output_str, file=kwargs.get("file", sys.stdout), end="")
print.indent = 0