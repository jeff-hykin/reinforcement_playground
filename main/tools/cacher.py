from super_hash import super_hash

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
    output = pickle.loads(bytes_in)
    return output

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

_lookup_table = {}
def cache(cache_folder="__pycache__", bust=False, no_pickle=False):
    # 
    # make cache folder
    # 
    import os
    import inspect
    # https://stackoverflow.com/questions/28021472/get-relative-path-of-caller-in-python
    try:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        directory = os.path.dirname(module.__file__)
    # if inside a repl (error =>) assume that the working directory is the path
    except AttributeError as error:
        directory = os.getcwd()
    
    folder_path = os.path.join(directory, cache_folder)
    os.makedirs(folder_path, exist_ok=True)
    
    def decorator_func(function_being_wrapped):
        function_hash = super_hash(function_being_wrapped)
        
        def wrapper(*args, **kwargs):
            arg_hash = super_hash(args)
            kwarg_hash = super_hash(kwargs)
            fingerprint = super_hash((function_hash, arg_hash, kwarg_hash))
            file_path = os.path.join(folder_path, str(fingerprint)+".quick_cache.pickle")
            # 
            # delete if needed
            # 
            if bust:
                del _lookup_table[fingerprint]
                try:
                    os.remove(file_path)
                except Exception as error:
                    pass
            should_refresh = not (hasattr(wrapper, "_refresh") and wrapper._refresh)
            if should_refresh:
                # 
                # cached in variable
                # 
                if fingerprint in _lookup_table:
                    print("loaded cached value")
                    return _lookup_table[fingerprint]
                
                # 
                # cached in file
                # 
                if os.path.isfile(file_path) and not no_pickle:
                    try:
                        _lookup_table[fingerprint] = large_pickle_load(file_path)
                        print("loaded cached value")
                    except Exception as error:
                        # clear corrupted data by deleting it
                        os.remove(file_path)
                
            # 
            # not cached
            # 
            if not os.path.isfile(file_path) and should_refresh:
                _lookup_table[fingerprint] = function_being_wrapped(*args, **kwargs)
                wrapper._refresh = False
                if not no_pickle:
                    large_pickle_save(_lookup_table[fingerprint], file_path)
                
            return _lookup_table[fingerprint]
        
        # 
        # add refresh helper
        # 
        def set_refresh():
            wrapper._refresh = True
        wrapper.refresh = set_refresh
        
        return wrapper
    return decorator_func
