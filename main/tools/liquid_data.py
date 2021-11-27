from tools.basics import flatten_once
from super_hash import super_hash
from super_map import LazyDict

class LazyList:
    def __init__(self, iterable, length=None):
        if isinstance(iterable, (tuple, list)):
            self.remaining = (each for each in (0,))
            next(self.remaining)
            self.memory = iterable
        else:
            self.remaining = (each for each in iterable)
            self.memory = []
        self._length = length
    
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
    
    def __json__(self):
        return [ each for each in self ]


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
    @classmethod
    def stats(cls, list_of_numbers: [float]) -> LazyDict:
        """
        Example:
            stats = LiquidData.stats([ 1, 4, 24.4, 5, 99 ])
            print(stats.min)
            print(stats.max)
            print(stats.range)
            print(stats.average)
            print(stats.median)
            print(stats.sum)
            print(stats.was_dirty)
            print(stats.cleaned)
            print(stats.count) # len(stats.cleaned)
        """
        import math
        numbers_as_tuple = tuple(list_of_numbers)
        original_length = len(numbers_as_tuple)
        # sort and filter bad values
        list_of_numbers = tuple(
            each
                for each in sorted(numbers_as_tuple)
                if not (each is None or each is math.nan)
        )
        # if empty return mostly bad values
        if len(list_of_numbers) == 0:
            return LazyDict(
                min=None,
                max=None,
                range=0,
                average=0,
                median=None,
                sum=0,
                was_dirty=(original_length != len(list_of_numbers)),
                cleaned=list_of_numbers,
                count=len(list_of_numbers),
            )
        # 
        # Calculate stats
        # 
        # TODO: make this more efficient by making a new class, and having getters 
        #       for each of these values, so that theyre calculated as-needed (and then cached)
        median = list_of_numbers[math.floor(len(list_of_numbers)/2.0)]
        minimum = list_of_numbers[0]
        maximum = list_of_numbers[-1]
        sum_total = 0
        for each in list_of_numbers:
            sum_total += each
        return LazyDict(
            min=minimum,
            max=maximum,
            range=minimum-maximum,
            average=sum_total/len(list_of_numbers),
            median=median,
            sum=sum_total,
            was_dirty=(original_length != len(list_of_numbers)),
            cleaned=list_of_numbers,
            count=len(list_of_numbers),
        )
    
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
    def bundles(self,):
        # FIXME: make a recursive/nested version of this
        return LazyList(
            LazyList(
                self.group_levels[-1][each_index] for each_index in each_bundle 
            )
                for each_bundle in self.group_levels[-2]
        )
    
    def compute(self,):
        for each in self.bundles:
            # forces all elments to be computed
            len(each)
        return self
    
    def __iter__(self):
        return (each for each in self.bundles)
    
    def only_keep_if(self, func):
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
                print(f'There was an error in {func}')
                print(err)
                error = err
                return {}
        
        # filter the bundles
        new_liquid.group_levels[-2] = [
            # similar to init, but with filter
            LazyList( item_index for item_index, record_index in enumerate(each_bundle) if func(self.group_levels[-1][record_index]))
                for each_bundle in self.bottom_bundles
        ]
        if error is not None:
            print(f'There was an issue in the lambda {func} when using .only_keep_if() on {self.__class__}')
            raise error
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
                print(f'There was an error in {func}')
                print(err)
                error = err
                return {}
        
        # first create the new mapped values
        mapped_values = LazyList(
            wrapped_function(self.group_levels[-1][each_record_index]) for each_record_index in flatten_once(self.group_levels[-2])
        )
        if error is not None:
            print(f'There was an issue in the lambda {func} when using .map() on {self.__class__}')
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
        # TODO: make this lazy by having every item be a generator of elements in the bundle, dynamically generated by a default_dict based on the key
        for each_bundle in new_liquid.group_levels[-2]:
            aggregated = defaultdict(lambda: [])
            for each_record_index in each_bundle:
                each_record = new_liquid.group_levels[-1][each_record_index]
                if each_record:
                    # for each key in each record
                    for each_key in each_record:
                        keys.add(each_key)
                    for each_key in keys:
                        aggregated[each_key].append(each_record.get(each_key, None))
                else:
                    print(f'Theres a pointer to index {each_record_index}, but when I look in the next group it doesnt exist')
            element_count = len(each_bundle)
            for each_key, each_value in aggregated.items():
                # add leading None's to make sure the length is right for each key
                aggregated[each_key] = ([None] * (element_count-len(each_value))) + aggregated[each_key]
            new_bundles.append(aggregated)
            
        
        # replace the old nested bundles with values (effectively making it the "records" level)
        new_liquid.group_levels[-2] = new_bundles
        # remove the old records level (-2 becomes records -3 becomes -2)
        del new_liquid.group_levels[-1]
        return new_liquid
