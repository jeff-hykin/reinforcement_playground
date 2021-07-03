import gdown
import os
import torch
temp_folder_path = f"{os.environ.get('PROJECTR_FOLDER')}/settings/.cache/common_format_datasets"

class SimpleDataset(torch.utils.data.Dataset):
    """
    When inheriting from this, make sure to define the following
        __init__(self, **kwargs):
            super(self.__class__, self).__init__(**kwargs)
        
        __len__(self):
        
        get_input(self, index):
        
        get_output(self, index):
    """
    @property
    def input_shape(self):
        input_1, _ = self[0]
        return tuple(input_1.shape)

    @property
    def output_shape(self):
        _, output_1 = self[0]
        return tuple(output_1.shape)
    
    def __init__(self, transform_input=None, transform_output=None):
        # save these for later
        self.transform_input  = transform_input
        self.transform_output = transform_output
        
    def __getitem__(self, index):
        # 
        # input
        # 
        an_input = self.get_input(index)
        if self.transform_input:
            an_input = self.transform_input(an_input)
        # 
        # output
        # 
        corrisponding_output = self.get_output(index)
        if self.transform_output:
            corrisponding_output = self.transform_output(corrisponding_output)
        
        # return
        return an_input, corrisponding_output

# FIXME: the images are corrupt for some reason
class Mnist(SimpleDataset):
    """
    file structure:
        $DATASET_FOLDER/
            mnist/
                img_0/
                    data.jpg
                    data.integer
                img_1/
                    data.jpg
                    data.integer
                ...
    
    """
    number_of_classes = 10
    _path = None
    @classmethod
    def get_path(cls, quiet=True):
        """
        returns the mnist folder, downloading the mnist data if needed
        """
        if cls._path is None:
            url = 'https://drive.google.com/uc?id=1im9_agM-hHl8dKQU5uWh054rScjPINtz' # jeff hykin's google drive copy of mnist
            output = f'{temp_folder_path}/mnist.zip'
            gdown.cached_download(url, output, postprocess=gdown.extractall, quiet=quiet)
            
            cls._path = temp_folder_path+"/mnist"
        return cls._path
        
    def __init__(self, **kwargs):
        super(Mnist, self).__init__(**kwargs)
        self.folder_path = self.get_path()
        
        from tools.file_system_tools import FS
        # ignore hidden files/folders (start with a .)
        self.ids = [ each for each in FS.list_folders(self.folder_path) if each[0] != '.' ]

    def __len__(self):
        return len(self.ids)
        
    def get_input(self, index):
        from torchvision.io import read_image
        item_folder =  self.folder_path +"/"+ self.ids[index]
        return read_image(item_folder+"/data.jpg")

    def get_output(self, index):
        from tools.file_system_tools import FS
        from tools.pytorch_tools import to_tensor
        import torch.nn.functional as F
        
        item_folder =  self.folder_path +"/"+ self.ids[index]
        return int(FS.read(item_folder+"/data.integer"))

mnist_dataset = Mnist()
