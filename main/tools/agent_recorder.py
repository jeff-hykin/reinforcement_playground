from tools.basics import large_pickle_load, large_pickle_save, to_pure
from tools.file_system_tools import FileSystem
from informative_iterator import ProgressBar
from super_map import Map

import os
import math
import json
import random

def read_json(path):
    import json
    with open(path, 'r') as in_file:
        return json.load(in_file)

def write_json(path, data):
    import json
    with open(path, 'w') as outfile:
        json.dump(data, outfile)

class AgentRecorder():
    def __init__(self, save_to):
        self.save_to = FileSystem.absolute_path(save_to)
        # create if doesnt exist
        FileSystem.ensure_is_folder(save_to)
        self.info_cache_path = save_to+"/.summary/info.json"
        self._size = None
        self._data_file_extension = None
        self._metadata_file_extension = None
        # try to replace the "None" values efficiently
        self._attempt_load_summary_from_cache()
    
    @property
    def size(self):
        if self._size == None:
            self._size = self.largest_index()
            if self._size is None:
                self._size = 0
            else:
                self._size += 1
        return self._size
    
    def save(self, observation, action):
        large_pickle_save((observation, action), self.save_to+f"/{self.size}{self._data_file_extension}")
        write_json(
            path=self.save_to+f"/{self.size}{self._metadata_file_extension}",
            data=to_pure(action),
        )
        self._size += 1
        # update summary data
        write_json(path=self.info_cache_path, data={
            "_size" : self._size,
            "_data_file_extension" : self._data_file_extension,
            "_metadata_file_extension" : self._metadata_file_extension,
        })
    
    def batch_sampler(self, batch_size, preprocessing):
        def sampler():
            remaining_indicies = set(self.indicies())
            total = math.floor(len(remaining_indicies) / batch_size)
            batch_index = -1
            while len(remaining_indicies) > batch_size:
                batch_index += 1
                entries = random.sample(remaining_indicies, k=batch_size)
                # remove the ones we just sampled
                remaining_indicies = remaining_indicies - set(entries)
                batch = []
                # load all the ones in the batch
                for each_index in entries:
                    batch.append(
                        (each_index, *large_pickle_load(self.save_to+f"/{each_index}{self._data_file_extension}"))
                    )
                # do any compression/decompression/augmentation stuff
                yield preprocessing(batch)
        sampler.batch_size = batch_size
        sampler.number_of_batches = math.floor( self.size / batch_size)
        return sampler
    
    def load_index(self, index):
        data = large_pickle_load(self.save_to+f"/{index}{self._data_file_extension}")
        metadata = read_json(path=self.save_to+f"/{index}{self._metadata_file_extension}")
        return metadata, data
    
    def load_data(self):
        for each_name in self.names():
            yield large_pickle_load(self.save_to+f"/{each_name}{self._data_file_extension}")

    def load_metadata(self):
        for each_name in self.names():
            yield read_json(path=self.save_to+f"/{each_name}{self._metadata_file_extension}")
    
    def create_batch_data(self, batch_name, batch_size, preprocessing=lambda batch:batch):
        # create folder for batch
        batch_path = f"{self.save_to}/{batch_name}"
        FileSystem.ensure_is_folder(batch_path)
        
        # build a index lookup based on the action
        print("building action index")
        samples_by_action_type = Map()
        for progress, metadata in ProgressBar(self.load_metadata(), iterations=self.size):
            if metadata not in samples_by_action_type:
                samples_by_action_type[metadata] = set()
            samples_by_action_type[metadata].add(progress.index)
        
        actions = list(samples_by_action_type[Map.Keys])
        print("starting batch creation")
        # this actually overshoots because its unknown how many iterations there will be
        max_number_of_batches = math.floor(self.size / batch_size)
        for progress, batch_index in ProgressBar(max_number_of_batches):
            batch = []
            # 
            # build batch
            # 
            while len(batch) < batch_size:
                # 
                # pick action
                # 
                which_action = random.choice(actions)
                remaining_indicies = samples_by_action_type[which_action]
                # 
                # pick index
                # 
                if len(remaining_indicies) == 0:
                    return
                index_of_element = random.choice(tuple(remaining_indicies))
                remaining_indicies.remove(index_of_element)
                # 
                # load index
                #
                batch.append(
                    (index_of_element, *large_pickle_load(self.save_to+f"/{index_of_element}{self._data_file_extension}"))
                )
            # 
            # preprocess batch
            # 
            # do any compression/decompression/augmentation stuff
            batch = preprocessing(batch)   
            # save it
            large_pickle_save(batch, f"{batch_path}/{batch_index}")
                
    @property
    def load_batch_data(self):
        dataset = self
        class Batch:
            def __init__(self, batch_name, *, epochs=1):
                batch_path = f"{dataset.save_to}/{batch_name}"
                batch_names = list(FileSystem.list_files(batch_path))
                
                self.batch_name        = batch_name
                self.number_of_epochs  = epochs
                self.batches_per_epoch = len(batch_names)
                self.length = self.batches_per_epoch * self.number_of_epochs
                self.epoch_index = -1
                self.batch_index = -1
                self.number = 0 # index + 1
                
                # create the reset indicies for the epochs
                total = self.batches_per_epoch
                self.reset_indicies = set()
                while total < self.length:
                    self.reset_indicies.add(total)
                    total += self.batches_per_epoch
                
                def sampler():
                    random.shuffle(batch_names)
                    self.epoch_index += 1
                    for self.batch_index, each in enumerate(batch_names):
                        self.number += 1
                        yield large_pickle_load(f'{batch_path}/{each}')
                        
                self.generator = sampler
                self.iterator = sampler()
            
            def __len__(self,):
                return self.length
            
            def __iter__(self):
                self.iterator = self.generator()
                return self

            def __next__(self):
                # if the end of an epoch, with more epochs to go
                if self.number in self.reset_indicies:
                    # repeat
                    self.iterator = self.generator()
                    
                return next(self.iterator)
        
        return Batch
        
    
    def names(self,):
        return ( each.split(".")[0] for each in FileSystem.list_files(self.save_to) )
    
    def indicies(self, ):
        # negative 1 is encase the folder is empty
        return ( int(each) for each in self.names() )
    
    def largest_index(self, ):
        indicies = tuple(self.indicies())
        if len(indicies) > 0:
            return max(indicies)
        else:
            return None
    
    def _attempt_load_summary_from_cache(self):
        self._size = None
        self._data_file_extension = None
        self._metadata_file_extension = None
        FileSystem.ensure_is_folder(FileSystem.dirname(self.info_cache_path))
        if FileSystem.is_file(self.info_cache_path):
            try:
                data = read_json(path=self.info_cache_path)
                self._size = data["_size"]
                self._data_file_extension = data["_data_file_extension"]
                self._metadata_file_extension = data["_metadata_file_extension"]
            except Exception as error:
                pass
        # if data corrupt or doesn't exist
        if not isinstance(self._metadata_file_extension, dict):
            self._size = self.size
            self._data_file_extension = ".data.pickle"
            self._metadata_file_extension = ".metadata.json"
            write_json(path=self.info_cache_path, data={
                "_size" : self._size,
                "_data_file_extension" : self._data_file_extension,
                "_metadata_file_extension" : self._metadata_file_extension,
            })
