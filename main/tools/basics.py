from collections import Counter # frequency count
import os
import sys
import math
import json
from time import time as now
import time

def reload():
    """
    reloads all imported modules
    (for debugging)
    """
    import sys
    import importlib
    for module in sys.modules.values():
        importlib.reload(module)

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


def recursively_map(an_object, function, is_key=False):
    from tools.basics import is_iterable
    
    
    # base case 1 (iterable but treated like a primitive)
    if isinstance(an_object, str):
        return_value = an_object
    # base case 2 (exists because of scalar numpy/pytorch/tensorflow objects)
    if hasattr(an_object, "tolist"):
        return_value = an_object.tolist()
    else:
        # base case 3
        if not is_iterable(an_object):
            return_value = an_object
        else:
            if isinstance(an_object, dict):
                return_value = { recursively_map(each_key, function, is_key=True) : recursively_map(each_value, function) for each_key, each_value in an_object.items() }
            else:
                return_value = [ recursively_map(each, function) for each in an_object ]
    
    # convert lists to tuples so they are hashable
    if is_iterable(return_value) and not isinstance(return_value, dict) and not isinstance(return_value, str):
        return_value = tuple(return_value)
    
    return function(return_value, is_key=is_key)

def to_pure(an_object, recursion_help=None):
    from tools.basics import is_iterable
    
    # 
    # infinte recursion prevention
    # 
    top_level = False
    if recursion_help is None:
        top_level = True
        recursion_help = {}
    class PlaceHolder:
        def __init__(self, id):
            self.id = id
        def eval(self):
            return recursion_help[key]
    object_id = id(an_object)
    # if we've see this object before
    if object_id in recursion_help:
        # if this value is a placeholder, then it means we found a child that is equal to a parent (or equal to other ancestor/grandparent)
        if isinstance(recursion_help[object_id], PlaceHolder):
            return recursion_help[object_id]
        else:
            # if its not a placeholder, then we already have cached the output
            return recursion_help[object_id]
    # if we havent seen the object before, give it a placeholder while it is being computed
    else:
        recursion_help[object_id] = PlaceHolder(object_id)
    
    parents_of_placeholders = set()
    
    # 
    # main compute
    # 
    return_value = None
    # base case 1 (iterable but treated like a primitive)
    if isinstance(an_object, str):
        return_value = an_object
    # base case 2 (exists because of scalar numpy/pytorch/tensorflow objects)
    elif hasattr(an_object, "tolist"):
        return_value = an_object.tolist()
    else:
        # base case 3
        if not is_iterable(an_object):
            return_value = an_object
        else:
            if isinstance(an_object, dict):
                return_value = {
                    to_pure(each_key, recursion_help) : to_pure(each_value, recursion_help)
                        for each_key, each_value in an_object.items()
                }
            else:
                return_value = [ to_pure(each, recursion_help) for each in an_object ]
    
    # convert iterables to tuples so they are hashable
    if is_iterable(return_value) and not isinstance(return_value, dict) and not isinstance(return_value, str):
        return_value = tuple(return_value)
    
    # update the cache/log with the real value
    recursion_help[object_id] = return_value
    #
    # handle placeholders
    #
    if is_iterable(return_value):
        # check if this value has any placeholder children
        children = return_value if not isinstance(return_value, dict) else [ *return_value.keys(), *return_value.values() ]
        for each in children:
            if isinstance(each, PlaceHolder):
                parents_of_placeholders.add(return_value)
                break
        # convert all the placeholders into their final values
        if top_level == True:
            for each_parent in parents_of_placeholders:
                iterator = enumerate(each_parent) if not isinstance(each_parent, dict) else each_parent.items()
                for each_key, each_value in iterator:
                    if isinstance(each_parent[each_key], PlaceHolder):
                        each_parent[each_key] = each_parent[each_key].eval()
                    # if the key is a placeholder
                    if isinstance(each_key, PlaceHolder):
                        value = each_parent[each_key]
                        del each_parent[each_key]
                        each_parent[each_key.eval()] = value
    
    # finally return the value
    return return_value


relative_path = lambda *filepath_peices : os.path.join(os.path.dirname(__file__), *filepath_peices)

def large_pickle_load(file_path):
    """
    This is for loading really big python objects from pickle files
    ~4Gb max value
    """
    import pickle
    import os
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)

def large_pickle_save(variable, file_path):
    """
    This is for saving really big python objects into a file
    so that they can be loaded in later
    ~4Gb max value
    """
    import pickle
    bytes_out = pickle.dumps(variable, protocol=4)
    max_bytes = 2**31 - 1
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


# just a self-made fix for unhashable builtin types
def hash_decorator(hash_function):
    import collections

    def is_iterable(thing):
        # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
        try:
            iter(thing)
        except TypeError:
            return False
        else:
            return True
            
    def make_hashable(value):
        type_of_value = type(value)
        output = None
        if type_of_value == str or type_of_value == frozenset:
            output = value
        elif type_of_value == set:
            output = frozenset([ make_hashable(each) for each in value ])
        elif type_of_value == dict:
            sorted_iterable = list(value.items())
            sorted_iterable.sort()
            output = tuple([ make_hashable(each) for each in sorted_iterable ])
        elif type_of_value == pd.core.frame.DataFrame:
            value_as_string = value.to_csv()
            output = hash(value_as_string)
        elif is_iterable(value):
            output = tuple([ make_hashable(each) for each in value ])
        else:
            output = value
        return output
        
    def wrapper(*args, **kwargs):
        try:
            return hash_function(*args, **kwargs)
        except:
            if len(args) == 1 and len(kwargs) == 0:
                hashable_argument = make_hashable(args[0])
                hashed_value = make_hashable(hashable_argument)
                return hashed_value
            return None
            
    return wrapper

def max_index(iterable):
    max_value = max(iterable)
    return to_pure(iterable).index(max_value)

# wrap the builtin hash function
hash = hash_decorator(hash)

# save loading times without brittle code
def auto_cache(function, *args, **kwargs):
    # 
    # create hash for arguments
    # 
    try:
        unique_hash = str(function.__name__)+"_"+str(hash(hash((args, kwargs))))
    except:
        unique_hash = None
    if type(unique_hash) != str:
        print(f"the arguments for {function.__name__} couldn't be auto cached")
        print("It probably contains some value that python doesn't know how to hash")
        print('args = ', args)
        print('kwargs = ', kwargs)
        print("running the function manually instead (failsafe)")
        return function(*args, **kwargs)
    
    # make the folders for the cache
    path_to_cache = relative_path("cache.nosync", f"{unique_hash}")
    ensure_folder(os.path.dirname(path_to_cache))

    # if the cache (for these arguments) exists, then just load it
    if os.path.exists(path_to_cache):
        return large_pickle_load(path_to_cache)
    # otherwise create it
    else:
        print(f"cache for {function.__name__} (with the current args) didn't exist")
        print("building cache now...")
        result = function(*args, **kwargs)
        try:
            large_pickle_save(result, path_to_cache)
            print("cache built")
        except:
            print(f"the result of {function.__name__} couldn't be auto cached")
            print("It probably contains some value that python doesn't know how to pickle")
            print("or the size of the output is larger than 4gb (less likely)")
            print('args = ', args)
            print('kwargs = ', kwargs)
            print("running the function manually instead (failsafe)")
        return result

        
import os
here = "os.path.dirname(__file__)"
if os.environ.get('PROJECTR_FOLDER', None):
    temp_folder = f"{os.environ.get('PROJECTR_FOLDER')}/settings/.cache/"
else:
    temp_folder = f"/tmp/"