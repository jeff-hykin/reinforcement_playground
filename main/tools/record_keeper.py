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
        if data == None: data = {}
        self.self = data
    
    def keys(self):
        return (each for each in tuple(self.parent.keys())+tuple(self.self.keys()))
    
    def values(self):
        return (each for each in tuple(self.parent.values())+tuple(self.self.values()))
    
    def items(self):
        return (each for each in tuple(self.parent.items())+tuple(self.self.items()))
    
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
        if key in self.parent:
            return self.parent[key]
        else:
            return self.self.get(key, None)
    
    def __setitem__(self, key, value):
        try:
            if not hasattr(self, "self") or not isinstance(self, dict):
                self.self = {}
            self.self[key] = value
        except Exception as error:
            print('hasattr(self, "self") = ', hasattr(self, "self"))
            print('list(self.__dict__.keys()) = ', list(self.__dict__.keys()))
    
    def __repr__(self,):
        copy = self.parent.copy()
        copy.update(self.self)
        return copy.__repr__()
    
    def get(self,*args,**kwargs):
        copy = self.parent.copy()
        copy.update(self.self)
        return copy.get(*args,**kwargs)

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
    def __init__(self, parent=None, collection=None, records=None):
        self.parent          = parent or {}
        self.kids            = []
        self.pending_record  = CustomInherit(parent=self.parent)
        self.collection      = collection
        self.records         = records or []
        if collection:
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
        print("Saving records")
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
                    "experiment_start_time": now(),
                },
            ),
            collection=self,
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


class LazyList:
    def __init__(self, iterable):
        if isinstance(iterable, (tuple, list)):
            self.remaining = (each for each in (0,))
            next(self.remaining)
            self.memory = iterable
        else:
            self.remaining = (each for each in iterable)
            self.memory = []
        self._length = None
    
    def __len__(self):
        if self._length == None:
            self.memory = tuple(self.memory) + tuple(self.remaining)
            self._length = len(self.memory)
        
        return self._length
    
    def __iter__(self):
        index = -1
        while True:
            index += 1
            try:
                # if already computed, just return it
                if index < len(self.memory):
                    yield self.memory[index]
                # if it isn't it memory then it must be the next element
                else:
                    the_next = next(self.remaining)
                    self.memory.append(the_next)
                    yield the_next
            except StopIteration:
                return
    
    def __getitem__(self, key):
        # allow negative indexing
        if key < 0:
            key = len(self) - key
        
        # if key is bigger than whats in memory
        while key+1 > len(self.memory):
            # use self iteration to grow
            # there might be a faster/bulk way to do this
            self.memory.append(next(self.remaining))
        
        return self.memory[key]
    
    def __setitem__(self, key, value):
        # allow negative indexing
        if key < 0:
            key = len(self) - key
        # make sure memory is at least this big
        self[key]
        # make sure memory is a list
        if type(self.memory) != list:
            self.memory = list(self.memory)
        self.memory[key] = value

class LiquidData():
    """
        from collections import defaultdict
        color_map = defaultdict(lambda: "#003AAZ")
        color_map.update({
            1: "#503A73",
            2: "#003973",
            3: "#003AAZ",
        })
        
        from statistics import mean as average
        LiquidData(records).only_keep_if(
                lambda each: each["training"],
            ).bundle_by(
                "experiment_number",
            # nested bundle
            ).bundle_by(
                "index",
            # convert lists-of-dictionaries into a dictionary-of-lists 
            ).aggregate(
            # average the loss over each index, within each experiment
            ).map(lambda each_iter_in_each_experiment: {
                **each_iter_in_each_experiment,
                "loss_1": average(each_iter_in_each_experiment["loss_1"]),
            # go back to only grouping by experiment number
            }).unbundle(
            # create one dict per experiment, with key-values having list-values
            ).aggregate(
            # for every experiment, add a label, a color, and extract out a list of x/y values
            ).map(lambda each: {
                "label": str(each["experiment_number"]),
                "backgroundColor": color_map[each["experiment_number"]],
                "color": color_map[each["experiment_number"]],
                "data": zip(each["index"], each["loss_1"])
            })
        
        Test:
                
            collection = ExperimentCollection("test1")
            r = collection.records
            a = z[ z['training'] == True ]
            b = a.groupby(["experiment_number", "index"], as_index=False)
            c = b.aggregate(func=tuple)
            d = c.groupby(["experiment_number"])

            from collections import defaultdict
            color_map = defaultdict(lambda: "#003AAZ")
            color_map.update({
                1: "#503A73",
                2: "#003973",
                3: "#003AAZ",
            })

            L = LiquidData(r)
            aa = L.only_keep_if(lambda each: each["training"] and each["loss_1"] is not None)
            bb = aa.bundle_by("experiment_number")
            cc = bb.bundle_by("index")
            dd = cc.aggregate()
            ee = dd.map(lambda each_iter_in_each_experiment: { "experiment_number": each_iter_in_each_experiment["experiment_number"][0], "index": each_iter_in_each_experiment["index"][0], "loss_1": average(each_iter_in_each_experiment["loss_1"]), })
            gg = ee.aggregate()
            hh = gg.map(lambda each: dict(index=each["index"], loss_1=each["loss_1"], experiment_number=each["experiment_number"][0]))
            ii = hh.map(lambda each: {
                    "label": str(each["experiment_number"]),
                    "backgroundColor": color_map[each["experiment_number"]],
                    "color": color_map[each["experiment_number"]],
                    "data": tuple(zip(each["index"], each["loss_1"]))
                })
    """
    def __init__(self, records=None, group_levels=None, internally_called=False):
        if internally_called:
            self.group_levels = group_levels
        else:
            # bottom level is always an iterable of some-kind of list-of-dictionaries
            # all other levels can be thought of as lists-of-lists-of-dictionaries
            # however, that "of-dictionaries" is actually lambda's, where each lambda points to a dictionary
            # (the lambdas act like pointers)
            self.group_levels = [ 
                # bundles
                [
                    # initially only one big bundle
                    LazyList(
                        index for index, _ in enumerate(records)
                    ),
                ],
                LazyList(records),
            ]
    
    @property
    def bottom(self,):
        return self.group_levels[-1]
    
    @property
    def bottom_bundles(self,):
        return self.group_levels[-2]
    
    @property
    def elements(self,):
        # FIXME: make this recursive/nested
        return LazyList(
            LazyList(
                self.group_levels[-1][each_index] for each_index in each_bundle 
            )
                for each_bundle in self.group_levels[-2]
        )
    
    def __iter__(self):
        return (each for each in self.elements)
    
    def only_keep_if(self, func):
        new_liquid = LiquidData(
            group_levels=list(self.group_levels),
            internally_called=True,
        )
        # filter the bundles
        new_liquid.group_levels[-2] = [
            # similar to init, but with filter
            LazyList( item_index for item_index, record_index in enumerate(each_bundle) if func(self.group_levels[-1][record_index]))
                for each_bundle in self.bottom_bundles
        ]
        return new_liquid
    
    def bundle_by(self, *keys):
        # find the number of unique values for those keys
        from collections import defaultdict
        new_liquid = LiquidData(
            group_levels=list(self.group_levels),
            internally_called=True,
        )
        def compute_sub_bundles(each_old_bundle):
            groups = defaultdict(lambda: [])
            for each_record_index in each_old_bundle:
                each_record = self.group_levels[-1][each_record_index]
                value_of_specified_keys = tuple(each_record.get(each_key, None) for each_key in keys)
                which_group = super_hash(value_of_specified_keys)
                groups[which_group].append(each_record_index)
            # sub-bundles
            return LazyList(each for each in groups.values())
        
        # list-of-bundles-of-lambdas-to-dictionary => list-of-bundles-of-bundles-of-lambdas-to-dictionary
        intermediate_list = [
            # a new bundle (bundle of bundles)
            compute_sub_bundles(each_old_bundle)
                for each_old_bundle in self.bottom_bundles
        ]
        # bundles of sub-bundles flattened out into just a bunch of sub-bundles
        new_bottom = LazyList(flatten_once(intermediate_list))
        # list-of-bundles-of-bundles-to-dictionary => list-of-bundles-of-lambdas-to-bundles-of-dictionary
        reconnected_old_bottom = []
        index_of_next_level = len(new_liquid.group_levels) - 2
        index_within_next_level = -1
        for each_bundle in intermediate_list:
            # make new bundles
            reconnected_old_bottom.append([])
            # each element in the new bundle is an index to a bundle in the next level
            for sub_bundle_index, _ in enumerate(each_bundle):
                index_within_next_level += 1
                reconnected_old_bottom[-1].append(index_within_next_level)
        
        # update the old level
        new_liquid.group_levels[-2] = reconnected_old_bottom
        # add the new level
        new_liquid.group_levels.insert(len(new_liquid.group_levels)-1, new_bottom)
        
        return new_liquid
    
    def unbundle(self):
        if len(self.group_levels) == 2:
            raise Exception('Tried to unbundle, but there were no bundles')
        new_liquid = LiquidData(
            group_levels=list(self.group_levels),
            internally_called=True,
        )
        old_before_bottom = self.group_levels[-3]
        bottom = self.bottom_bundles
        new_before_bottom = LazyList(
            # each bundle, it is still a bundle but the sub-bundles have been flattened
            flatten_once(LazyList(
                # each_value is an index to a sub_bundle (which is in a different level)
                # (e.g. below is a sub-bundle)
                bottom[each_value]
                    for each_value in each_bundle
            ))
                for each_bundle in old_before_bottom
        )
        # remove the bottom
        del new_liquid.group_levels[-2]
        # replace old before-bottom with the new flattened before-bottom
        new_liquid.group_levels[-2] = new_before_bottom
        return new_liquid
    
    def map(self, func):
        new_liquid = LiquidData(
            group_levels=list(self.group_levels),
            internally_called=True,
        )
        error = None
        def wrapped_function(record):
            nonlocal error
            try:
                return func(record)
            except Exception as err:
                error = err
                return None
        
        # first create the new mapped values
        mapped_values = LazyList(
            func(self.group_levels[-1][each_record_index]) for each_record_index in flatten_once(self.group_levels[-2])
        )
        if error:
            print(f'There was an issue when using .map() on {self.__class__}')
            raise error
        # replace the old values with the new values
        new_liquid.group_levels[-1] = mapped_values
        # then update the indicies of the bundles
        new_indicies = (each_index for each_index, _ in enumerate(mapped_values))
        new_liquid.group_levels[-2] = LazyList(
            LazyList(
                next(new_indicies)
                    for each_record_index in each_bundle
            )
                for each_bundle in self.group_levels[-2]
        )
        return new_liquid
    
    def aggregate(self):
        new_liquid = LiquidData(
            group_levels=list(self.group_levels),
            internally_called=True,
        )
        
        # ensure there is at least one bundle
        if len(new_liquid.group_levels) < 3:
            new_liquid.group_levels.insert(0, [
                # one bundle, that contains one element for every already-existing bundle
                LazyList(each_index for each_index, _ in enumerate(self.group_levels[0]))
            ])
        
        # this is one of the only parts that I don't think would be very effecitve as 100% a generator
        # it does a single pass instead of (as a generator) O(n) * number of dictionary keys
        #   # part of an alternative/iterative method
        #   # {
        #   #     each_key: LazyList(
        #   #         self.group_levels[-1][each_record_index].get(each_key, None)
        #   #             for each_record_index in each_bundle
        #   #     )
        #   #         for each_key in keys 
        #   # }
        #   #         for each_bundle in new_liquid.group_levels[-2]
        from collections import defaultdict
        new_bundles = []
        keys = set()
        for each_bundle in new_liquid.group_levels[-2]:
            aggregated = defaultdict(lambda: [])
            for each_record_index in each_bundle:
                each_record = self.group_levels[-1][each_record_index]
                # for each key in each record
                for each_key in each_record:
                    keys.add(each_key)
                for each_key in keys:
                    aggregated[each_key].append(each_record.get(each_key, None))
            new_bundles.append(aggregated)
            
        
        # replace the old nested bundles with values (effectively making it the "records" level)
        new_liquid.group_levels[-2] = new_bundles
        # remove the old records level (-2 becomes records -3 becomes -2)
        del new_liquid.group_levels[-1]
        return new_liquid
