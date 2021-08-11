#%%
from super_hash import super_hash
all_records = []
class RecordKeeper():
    def __init__(self, *args, **kwargs):
        class _SelfKey: pass
        
        self.parent = {"$ancestors":[]}
        self.parent.update(kwargs)
        if len(args) > 0 and isinstance(args[0], dict): self.parent.update(args[0])
        self.parent["$ancestors"] += self.parent
        self.current_record = None
        self._SelfKey = _SelfKey
        self.parent[self._SelfKey] = id(self)
    
    def parent_should_include(self, **kwargs):
        # add it to the current element
        self.parent.update(kwargs)
        return self
    
    def this_record_includes(self, **kwargs):
        if self.current_record is None:
            self.start_next_record()
        # add it to the current element
        self.current_record.update(kwargs)
        return self
    
    def start_next_record(self):
        # when adding a child, always have a link back to the parent data 
        # (this uses the ..., which is a valid/normalish value in python)
        self.current_record = {
            (...): self.parent,
        }
        all_records.append(self.current_record)
            
    def sub_record_keeper(self, **kwargs):
        return RecordKeeper({**self.parent, **kwargs, "$ancestors": self.parent["$ancestors"]})
    
    def __len__(self):
        return len(tuple(each_record for each_record in all_records if self._SelfKey in each_record[...]))
        
    def __repr__(self):
        size = len(self)
        parent = { each_key: each_value for each_key, each_value in self.parent.items() if not (type(each_key) != str or each_key == "$ancestors") }
        parent["__id__"] = self.id
        return f"""Parent: {parent}\n# of records: {size}"""
    
    def __iter__(self):
        return (each_record for each_record in all_records if self._SelfKey in each_record[...])
    
    def __getitem__(self, key):
        return self.current_record.get(key, None)
        
    def __setitem__(self, key, value):
        self.this_record_includes(**{key: value})
    
    @property
    def id(self):
        return super_hash(self.parent)
    
    def save(self, path):
        import json
        output = []
        for each_record in self:
            # combine parent data into element
            item = { **each_record[...], **dict(each_record), }
            # hash the ancestors
            item["$ancestors"] = [ super_hash(each) for each in item["$ancestors"] ]
            # remove un-json-able keys
            item  = { key : value for key, value in item.items() if type(key) == str }
            output.append(item)
        # save to file
        import os
        from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext, relpath
        os.makedirs(dirname(path), exist_ok=True)
        with open(path, 'w') as outfile:
            json.dump(output, outfile)
#%%