import gdown
import os
import torch
temp_folder_path = f"{os.environ.get('PROJECTR_FOLDER')}/settings/.cache/common_format_datasets"

class Mnist(torch.utils.data.Dataset):
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
    @classmethod
    def get_path(cls, quiet=False):
        """
        returns the mnist folder, downloading the mnist data if needed
        """
        url = 'https://drive.google.com/uc?id=1im9_agM-hHl8dKQU5uWh054rScjPINtz' # jeff hykin's google drive copy of mnist
        output = f'{temp_folder_path}/mnist.zip'
        gdown.cached_download(url, output, postprocess=gdown.extractall, quiet=quiet)
        
        return temp_folder_path+"/mnist"
        
    def __init__(self, transform_input=None, transform_output=None):
        self.transform_input  = transform_input
        self.target_transform = target_transform
        self.folder_path = Mnist.get_path()
        
        from tools.defaults import FS
        # ignore hidden files/folders (start with a .)
        self.ids = [ each for each in FS.list_folders(folder_path) if each[0] != '.' ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        item_folder =  self.folder_path +"/"+ self.ids[index]
        # 
        # input
        # 
        from torchvision.io import read_image
        an_input = read_image(item_folder+"/data.jpg")
        if self.transform_input:
            an_input = self.transform_input(an_input)
        # 
        # output
        # 
        from tools.defaults import FS
        corrisponding_output = int(FS.read(item_folder+"/data.integer"))
        if self.transform_output:
            corrisponding_output = self.transform_output(corrisponding_output)
        
        # return
        return an_input, corrisponding_output

