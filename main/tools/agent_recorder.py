from tools.basics import large_pickle_load, large_pickle_save, to_pure
from tools.file_system_tools import FileSystem
from tools.progress_bar import ProgressBar
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
        
        # FIXME uniformly random sample from action space
        
        remaining_indicies = set(self.indicies())
        total = math.floor(len(remaining_indicies) / batch_size)
        batch_index = -1
        for each in ProgressBar(range(total)):
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
            batch = preprocessing(batch)
            # save it
            large_pickle_save(batch, f"{batch_path}/{batch_index}")
    
    def load_batch_data(self, batch_name):
        batch_path = f"{self.save_to}/{batch_name}"
        batch_names = FileSystem.list_files(batch_path)
        self.number_of_batches = len(batch_names)
        for each in batch_names:
            yield large_pickle_load(f'{batch_path}/{each}')
    
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
