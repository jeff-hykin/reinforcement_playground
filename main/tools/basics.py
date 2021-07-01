def product(iterable):
    from functools import reduce
    import operator
    return reduce(operator.mul, iterable, 1)

def is_iterable(thing):
    # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    try:
        iter(thing)
    except TypeError:
        return False
    else:
        return True

def flatten(value):
    flattener = lambda *m: (i for n in m for i in (flattener(*n) if is_iterable(n) else (n,)))
    return list(flattener(value))

import collections.abc
def merge(old_value, new_value):
    # if not dict, see if it is iterable
    if not isinstance(new_value, collections.abc.Mapping):
        if is_iterable(new_value):
            new_value = { index: value for index, value in enumerate(new_value) }
    
    # if still not a dict, then just return the current value
    if not isinstance(new_value, collections.abc.Mapping):
        return new_value
    # otherwise get recursive
    else:
        # if not dict, see if it is iterable
        if not isinstance(old_value, collections.abc.Mapping):
            if is_iterable(old_value):
                old_value = { index: value for index, value in enumerate(old_value) }
        # if still not a dict
        if not isinstance(old_value, collections.abc.Mapping):
            # force it to be one
            old_value = {}
        
        # override each key recursively
        for key, updated_value in new_value.items():
            old_value[key] = merge(old_value.get(key, {}), updated_value)
        
        return old_value

def bundle(iterable, bundle_size):
    next_bundle = []
    for each in iterable:
        next_bundle.append(each)
        if len(next_bundle) == bundle_size:
            yield tuple(next_bundle)
            next_bundle = []
    # return any half-made bundles
    if len(next_bundle) > 0:
        yield tuple(next_bundle)

import os
here = "os.path.dirname(__file__)"
if os.environ.get('PROJECTR_FOLDER', None):
    temp_folder = f"{os.environ.get('PROJECTR_FOLDER')}/settings/.cache/"
else:
    temp_folder = f"/tmp/"