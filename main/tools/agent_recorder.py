from tools.basics import large_pickle_load, large_pickle_save, to_pure
from tools.file_system_tools import FileSystem
import os
import math
import json
import random

class AgentRecorder():
    def __init__(self, save_to):
        self.save_to = FileSystem.absolute_path(save_to)
        # create if doesnt exist
        FileSystem.ensure_is_folder(save_to)
        # find where we left off
        self.previous_index = self.largest_index()
        if self.previous_index is None:
            self.previous_index = -1
        
        self.data_file_extension = ".data.pickle"
        self.metadata_file_extension = ".metadata.json"
    
    def save(self, observation, action):
        self.previous_index += 1
        large_pickle_save((observation, action), self.save_to+f"/{self.previous_index}{self.data_file_extension}")
        with open(self.save_to+f"/{self.previous_index}{self.metadata_file_extension}", 'w') as outfile:
            json.dump(to_pure(action), outfile)
    
    def load_data(self):
        for each_name in self.names():
            yield large_pickle_load(self.save_to+f"/{each_name}{self.data_file_extension}")

    def load_metadata(self):
        for each_name in self.names():
            with open(self.save_to+f"/{each_name}{self.metadata_file_extension}", 'r') as in_file:
                yield json.load(in_file)
    
    def create_batch_data(self, batch_name, batch_size, preprocessing=lambda each:each):
        # create folder for batch
        batch_path = f"{self.save_to}/{batch_name}"
        FileSystem.ensure_is_folder(batch_path)
        
        # FIXME uniformly random sample from action space
        
        remaining_indicies = set(self.indicies())
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
                    large_pickle_load(self.save_to+f"/{each_index}{self.data_file_extension}")
                )
            # do any compression/decompression/augmentation stuff
            batch = preprocessing(batch)
            # save it
            print(f'saving batch {batch_index}')
            large_pickle_save(batch, f"{batch_path}/{batch_index}")
    
    def load_batch_data(self, batch_name):
        # create folder for batch
        batch_path = f"{self.save_to}/{batch_name}"
        batch_names = FS.list_files(batch_path)
        for each in batch_names:
            yield large_pickle_load(each)
    
    def names(self,):
        file_pieces = tuple( FileSystem.path_pieces(each) for each in FileSystem.list_files(self.save_to) )
        names = tuple( file_name for *folders, file_name, file_extension in file_pieces )
        return names
    
    def indicies(self, ):
        # negative 1 is encase the folder is empty
        return tuple( int(each) for each in self.names() )
    
    def largest_index(self, ):
        indicies = self.indicies()
        if len(indicies) > 0:
            return max(indicies)
        else:
            return None
