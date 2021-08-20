#%%
from super_hash import super_hash
from tools.basics import large_pickle_load, large_pickle_save, attempt

class CustomInherit(dict):
    def __init__(self, *, parent, data=None):
        self.parent = parent
        if data == None: data = {}
        for each_key, each_value in data.items():
            self[each_key] = each_value
    
    @property
    def dict(self):
        core = super()
        parent_data_copy = dict(self.parent)
        for each_key, each_value in core.items():
            parent_data_copy[each_key] = each_value
        return parent_data_copy
    
    def keys(self):
        return self.dict.keys()
    
    def values(self):
        return self.dict.values()
    
    def items(self):
        return self.dict.items()
    
    @property
    def ancestors(self):
        current = self
        ancestors = []
        while hasattr(current, "parent") and current != current.parent:
            ancestors.append(current.parent)
            current = current.parent
        return ancestors
    
    def __len__(self):
        return len(self.dict)
    
    def __iter__(self):
        return self.dict.keys()
    
    def __getitem__(self, key):
        return self.dict.get(key, None)
    
    def __setitem__(self, key, value):
        self.update({key: value})
    
    def __repr__(self,):
        return self.dict.__repr__()


class RecordKeeper():
    def __init__(self, parent, file_path, all_records, all_record_keepers):
        self.parent         = parent
        self.kids           = []
        self.file_path      = file_path
        self.all_records    = all_records
        self.current_record = None
        self.record_keepers = all_record_keepers
        self.record_keepers[super_hash(self)] = self
    
    def merge(self, **kwargs):
        if self.current_record is None:
            # when adding a record, always have a link back to the parent data 
            self.current_record = CustomInherit(parent=self.parent)
            self.all_records.append(self.current_record)
        # add it to the current element
        self.current_record.update(kwargs)
        return self
    
    def start_next_record(self):
        # delete the existing record and a new one will be created automatically as soon as data is added
        self.current_record = None
    
    def sub_record_keeper(self, **kwargs):
        grandparent = self.parent
        kid = RecordKeeper(
            parent=CustomInherit(parent=grandparent, data=kwargs),
            file_path=self.file_path,
            all_records=self.all_records,
            all_record_keepers=self.record_keepers,
        )
        self.kids.append(kid)
        return kid
    
    @property
    def info(self):
        return self.parent
        
    @property
    def ancestors(self):
        return [ self.parent, *self.parent.ancestors ]
    
    def __iter__(self):
        return (each for each in self.all_records if self.parent in each.ancestors)
    
    def __len__(self):
        # apparently this is the fastest way (no idea why converting to tuple is faster than using reduce)
        return len(tuple((each for each in self)))
    
    def __hash__(self):
        return super_hash({ "CustomInherit": self.parent })
        
    @property
    def records(self):
        return [ each for each in self ]
    
    def __repr__(self):
        size = len(self)
        return f"""Parent: {self.parent}\n# of records: {size}"""
    
    def __getitem__(self, key):
        return self.current_record.get(key, None)
    
    def __setitem__(self, key, value):
        self.merge(**{key: value})

#%%

# 
# create a "with" object
# 
class Experiment(object):
    def __init__(self, experiment, experiment_parent, record_keepers, file_path, collection_notes, records, collection_name):
        self.experiment        = experiment       
        self.experiment_parent = experiment_parent
        self.record_keepers    = record_keepers
        self.file_path         = file_path
        self.collection_notes  = collection_notes
        self._records           = records
        self.collection_name   = collection_name
    
    def __enter__(self):
        return self.experiment
    
    def __exit__(self, _, error, traceback):
        # mutate the root one based on having an error or not
        no_error = error is None
        if no_error:
            self.experiment_parent.info["had_error"] = False
            self.experiment_parent.info["error_number"] = 0
        
        # refresh the all_record_keepers dict
        # especially after mutating the self.experiment_parent.info
        # (this ends up acting like a set, but keys are based on mutable values)
        self.record_keepers = { super_hash(each_value) : each_value for each_value in self.record_keepers.values() }
        
        # 
        # save to file
        # 
        # ensure folder exists
        import os;os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        data = (self.collection_notes, self.experiment_parent.info, self.record_keepers, self._records)
        large_pickle_save(data, self.file_path)
        
        # re-throw the error
        if not no_error:
            print(f'There was an error when adding experiment to collection: "{self.collection_name}"')
            print(f'However, the partial experiment data was saved')
            experiment_number = self.experiment_parent.info["experiment_number"]
            error_number = self.experiment_parent.info["error_number"]
            print(f'This happend on:\n    dict(experiment_number={experiment_number}, error_number={error_number})')
            raise error

class ExperimentCollection:
    """
    Example:
        collection = ExperimentCollection("test1") # filepath 
        with collection.new_experiment() as record_keeper:
            model1 = record_keeper.sub_record_keeper(model="model1")
            model2 = record_keeper.sub_record_keeper(model="model2")
            model_1_losses = model1.sub_record_keeper(training=True)
            from random import random, sample, choices
            for each in range(1000):
                model_1_losses["index"] = each
                model_1_losses["loss_1"] = random()
                model_1_losses.start_next_record()

        collection.records[0]
        collection.records[-1]
    """
    def __init__(self, collection, records=None, extension=".pkl"):
        self.file_path              = collection+extension
        self.experiment             = None
        self.collection_name        = ""
        self.collection_notes       = {}
        self.experiment_parent      = None
        self._records               = records or []
        self.record_keepers         = {}
        self.prev_experiment_parent_info = None
        
        import os
        self.file_path = os.path.abspath(self.file_path)
        self.collection_name = os.path.basename(self.file_path)[0:-len(extension)]
    
    def load(self):
        # 
        # load from file
        # 
        import os
        self.prev_experiment_parent_info = dict(experiment_number=0, error_number=0, had_error=False)
        if not self._records and self.file_path:
            try: self.collection_notes, self.prev_experiment_parent_info, self.record_keepers, self._records = large_pickle_load(self.file_path)
            except: print(f'Will creaete new experiment collection: {self.collection_name}')
    
    def where(self, only_keep_if=None, exist=None):
        # TODO: add group, x value, y value mappers
        
        # "exists" lambda
        if exist is None: exist = []
        required_keys = set(exist)
        required_keys_exist = lambda each: len(required_keys & set(each.keys())) == len(required_keys) 
        if len(exist) == 0: required_keys_exist = lambda each: True
        
        # "only_keep_if" lambda
        if only_keep_if is None: only_keep_if = lambda each: True
        
        # combined
        the_filter = lambda each: only_keep_if(each) and required_keys_exist(each)
        # load if needed
        if not self._records:
            self.load()
        
        return (each for each in self._records if the_filter(each))
            
    def new_experiment(self, **experiment_info):
        if len(experiment_info) == 0: experiment_info = None
        
        # 
        # load from file
        # 
        self.load()
        
        # add basic data to the experiment
        # there are 3 levels:
        # - self.collection_notes (root)
        # - self.experiment_parent
        # - self.experiment
        self.experiment_parent = RecordKeeper(
            parent=CustomInherit(
                parent=self.collection_notes,
                data={
                    "experiment_number": self.prev_experiment_parent_info["experiment_number"] + 1 if not self.prev_experiment_parent_info["had_error"] else self.prev_experiment_parent_info["experiment_number"],
                    "error_number": self.prev_experiment_parent_info["error_number"]+1,
                    "had_error": True,
                },
            ),
            file_path=self.file_path,
            all_records=self._records,
            all_record_keepers=self.record_keepers,
        )
        # create experiment record keeper
        if experiment_info is None:
            self.experiment = self.experiment_parent
        else:
            self.experiment = self.experiment_parent.sub_record_keeper(**experiment_info)
        return Experiment(
            experiment=self.experiment,
            experiment_parent=self.experiment_parent,
            record_keepers=self.record_keepers,
            file_path=self.file_path,
            collection_notes=self.collection_notes,
            records=self._records,
            collection_name=self.collection_name,
        )
    
    def add_notes(self, notes, records=None, extension=".pkl"):
        import os
        file_path = os.path.abspath(collection+extension)
        collection_name = os.path.basename(file_path)[0:-len(extension)]
        # default values
        collection_notes = {}
        prev_experiment_parent_info = dict(experiment_number=0, error_number=0, had_error=False)
        record_keepers = {}
        records = records or []
        # attempt load
        try: collection_notes, prev_experiment_parent_info, record_keepers, records = large_pickle_load(file_path)
        except: print(f'Will creaete new experiment collection: {collection_name}')
        # merge data
        collection_notes.update(notes)
        # save updated data
        data = (collection_notes, prev_experiment_parent_info, record_keepers, records)
        large_pickle_save(data, file_path)
        
#%%

# %%
main.save()
# %%
