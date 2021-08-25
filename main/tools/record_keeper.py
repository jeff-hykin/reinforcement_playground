#%%
from super_hash import super_hash
from tools.basics import large_pickle_load, large_pickle_save, attempt, flatten_once
from time import time as now

#%%
class CustomInherit(dict):
    def __init__(self, *, parent, data=None):
        self.parent = parent
        if not isinstance(self.parent, dict):
            raise Exception('for CustomInherit(), parent needs to be a dict')
        self._self = data or {}
    
    @property
    def self(self):
        if not hasattr(self, "_self"):
            self._self = {}
        return self._self
    
    def keys(self):
        self_keys = self.self.keys()
        for each_key in self_keys:
            yield each_key
        self_keys = set(self_keys)
        for each_key in self.parent.keys():
            if each_key not in self_keys:
                yield each_key
    
    def values(self):
        self_keys = self.self.keys()
        for each_key, each_value in self.self.items():
            yield each_value
        self_keys = set(self_keys)
        for each_key, each_value in self.parent.items():
            if each_key not in self_keys:
                yield each_value
    
    def items(self):
        self_keys = self.self.keys()
        for each_key, each_value in self.self.items():
            yield (each_key, each_value)
        self_keys = set(self_keys)
        for each_key, each_value in self.parent.items():
            if each_key not in self_keys:
                yield (each_key, each_value)
    
    @property
    def ancestors(self):
        current = self
        ancestors = []
        while hasattr(current, "parent") and current != current.parent:
            ancestors.append(current.parent)
            current = current.parent
        return ancestors
    
    def __len__(self):
        return len(tuple(self.keys()))
    
    def __iter__(self):
        return (each for each in self.keys())
    
    def __contains__(self, key):
        return key in self.parent or key in self.self
        
    def __getitem__(self, key):
        if key in self.self:
            return self.self[key]
        else:
            return self.parent.get(key, None)
    
    def __setitem__(self, key, value):
        self.self[key] = value
    
    def __repr__(self,):
        copy = self.parent.copy()
        copy.update(self.self)
        return copy.__repr__()
    
    def get(self,*args,**kwargs):
        copy = self.parent.copy()
        copy.update(self.self)
        return copy.get(*args,**kwargs)
    
    def copy(self,*args,**kwargs):
        copy = self.parent.copy()
        copy.update(self.self)
        return copy.copy(*args,**kwargs)
    
    def __getstate__(self):
        return {
            "_self": self.self,
            "parent": self.parent,
        }
    
    def __setstate__(self, state):
        self._self = state["_self"]
        self.parent = state["parent"]
    
    def __json__(self):
        copy = self.parent.copy()
        copy.update(self.self)
        return copy

class RecordKeeper():
    """
        Example:
            record_keeper = RecordKeeper(
                parent=CustomInherit(
                    parent={},
                    data={
                        "experiment_number": 1,
                        "error_number": 0,
                        "had_error": True,
                    },
                ),
                collection=collection
            )
            a = record_keeper.sub_record_keeper(hi=10)
            a.hi # returns 10
    """
    def __init__(self, parent=None, collection=None, records=None, file_path=None):
        self.parent          = parent or {}
        self.file_path       = file_path
        self.kids            = []
        self.pending_record  = CustomInherit(parent=self.parent)
        self.collection      = collection
        self.records         = records or []
        if not isinstance(self.parent, dict):
            raise Exception('Parent needs to be a dict')
        self.setup()
    
    def setup(self):
        if self.collection is not None:
            self.get_records = lambda: self.collection.records
            self.add_record  = lambda each: self.collection.add_record(each)
        else:
            self.get_records = lambda: self.records
            self.add_record  = lambda each: self.records.append(each)
    
    def commit_record(self,*, additional_info=None):
        # finalize the record
        additional_info and self.pending_record.update(additional_info)
        # save a copy to disk
        self.add_record(self.pending_record)
        # start a new clean record
        self.pending_record = CustomInherit(parent=self.parent)
        
    def sub_record_keeper(self, **kwargs):
        grandparent = self.parent
        kid = RecordKeeper(
            parent=CustomInherit(parent=grandparent, data=kwargs),
            collection=self.collection,
            records=self.records,
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
        return (each for each in self.get_records() if self.parent in each.ancestors)
    
    def __len__(self):
        # apparently this is the fastest way (no idea why converting to tuple is faster than using reduce)
        return len(tuple((each for each in self)))
    
    def __hash__(self):
        return super_hash({ "CustomInherit": self.parent })
        
    def __repr__(self):
        size = len(self)
        return f"""Parent: {self.parent}\n# of records: {size}"""
    
    def __getitem__(self, key):
        if self.pending_record is not None:
            # current_record inherits from parent
            return self.pending_record[key]
        else:
            return self.parent.get(key, None)
    
    def __setitem__(self, key, value):
        self.parent[key] = value
    
    def __getattr__(self, key):
        return self[key]
    
    def copy(self,*args,**kwargs):
        return self.pending_record.copy(*args,**kwargs)
    
    def items(self, *args, **kwargs):
        return self.pending_record.items(*args, **kwargs)
    
    def keys(self, *args, **kwargs):
        return self.pending_record.keys(*args, **kwargs)
    
    def values(self, *args, **kwargs):
        return self.pending_record.values(*args, **kwargs)
    
    def __getstate__(self):
        return (self.parent, self.file_path, self.kids, self.pending_record, self.records)
    
    def __setstate__(self, state):
        self.parent, self.file_path, self.kids, self.pending_record, self.records = state
        self.collection = None
        if self.file_path is not None:
            self.collection = globals().get("_ExperimentCollection_register",{}).get(self.file_path, None)
        self.setup()
        if not isinstance(self.parent, dict):
            raise Exception('Parent needs to be a dict')

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
        self.experiment_parent.info["experiment_end_time"] = now()
        self.experiment_parent.info["experiment_duration"] = self.experiment_parent.info["experiment_end_time"] - self.experiment_parent.info["experiment_start_time"]
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
        print("Saving "+str(len(self._records))+" records")
        large_pickle_save(data, self.file_path)
        print("Records saved to: " + self.file_path)
        
        # re-throw the error
        if not no_error:
            print(f'There was an error when running an experiment. Experiment collection: "{self.collection_name}"')
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
                model_1_losses.pending_record["index"] = each
                model_1_losses.pending_record["loss_1"] = random()
                model_1_losses.commit_record()
        
        
        experiment_numbers = range(max(each["experiment_number"] for each in collection.records))
        groups = Morph.each(
            data=collection.records,
            if_keys_exist=["loss_1"],
            add_to_groups_if={
                "train": lambda each:each["train"]==True,
                "test": lambda each:not (each["train"]==True),
            },
            remorph=dict(
                add_to_groups_if={
                    experiment_number : (lambda each_record: each_record["experiment_number"] == experiment_number)
                        for experiment_number in experiment_numbers 
                },
                average={
                    "
                }
            )
        )
    """
    
    # TODO: make it so that Experiments uses database with detached/reattached pickled objects instead of a single pickle file
    
    def __init__(self, file_path, records=None, extension=".pickle"):
        self.file_path              = file_path+extension
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
        
        # when a record_keeper is seralized, it shouldn't contain a copy of the experiment collection and every single record
        # it really just needs its parents/children
        # however, it still needs a refernce to the experiment_collection to get access to all the records
        # so this register is used as a way for it to reconnect, based on the file_path of the collection
        register = globals()["_ExperimentCollection_register"] = globals().get("_ExperimentCollection_register", {})
        register[self.file_path] = self
        
        self.prev_experiment_parent_info = dict(experiment_number=0, error_number=0, had_error=False)
        if not self._records and self.file_path:
            if os.path.isfile(self.file_path):
                self.collection_notes, self.prev_experiment_parent_info, self.record_keepers, self._records = large_pickle_load(self.file_path)
            else:
                print(f'Will create new experiment collection: {self.collection_name}')
    
    def ensure_loaded(self):
        if self.prev_experiment_parent_info == None:
            self.load()
    
    def add_record(self, record):
        self.ensure_loaded()
        self._records.append(record)
    
    def where(self, only_keep_if=None, exist=None, groups=None, extract=None):
        # "exists" lambda
        if exist is None: exist = []
        required_keys = set(exist)
        required_keys_exist = lambda each: len(required_keys & set(each.keys())) == len(required_keys) 
        if len(exist) == 0: required_keys_exist = lambda each: True
        
        # "only_keep_if" lambda
        if only_keep_if is None: only_keep_if = lambda each: True
        
        # "extract" lambda
        if extract is None: extract = lambda each: each
        
        # combined
        the_filter = lambda each: only_keep_if(each) and required_keys_exist(each)
        # load if needed
        if not self._records:
            self.load()
        
        # TODO: add group, x value, y value mappers
        if groups is None:
            return (extract(each) for each in self._records if the_filter(each))
        else:
            group_finders = groups
            groups = { each: [] for each in group_finders }
            for each_record in self._records:
                if the_filter(each_record):
                    extracted_value = extract(each_record)
                    for each_group_name, each_group_finder in group_finders.items():
                        if each_group_finder(each_record):
                            groups[each_group_name].append(extracted_value)
            
            return groups
    
    def new_experiment(self, **experiment_info):
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
            collection=self,
            parent=CustomInherit(
                parent=self.collection_notes,
                data={
                    "experiment_number": self.prev_experiment_parent_info["experiment_number"] + 1 if not self.prev_experiment_parent_info["had_error"] else self.prev_experiment_parent_info["experiment_number"],
                    "error_number": self.prev_experiment_parent_info["error_number"]+1,
                    "had_error": True,
                    "experiment_start_time": now(),
                },
            ),
        )
        # create experiment record keeper
        if len(experiment_info) == 0:
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
    
    def __len__(self,):
        self.ensure_loaded()
        return len(self._records)
    
    @property
    def records(self):
        self.ensure_loaded()
        return self._records
    
    def add_notes(self, notes, records=None, extension=".pickle"):
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
