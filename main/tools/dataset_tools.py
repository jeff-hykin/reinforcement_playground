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
    number_of_classes = 10
    
    @classmethod
    def get_path(cls, quiet=True):
        """
        returns the mnist folder, downloading the mnist data if needed
        """
        url = 'https://drive.google.com/uc?id=1im9_agM-hHl8dKQU5uWh054rScjPINtz' # jeff hykin's google drive copy of mnist
        output = f'{temp_folder_path}/mnist.zip'
        gdown.cached_download(url, output, postprocess=gdown.extractall, quiet=quiet)
        
        return temp_folder_path+"/mnist"
        
    def __init__(self, transform_input=None, transform_output=None):
        self.transform_input  = transform_input
        self.transform_output = transform_output
        self.folder_path = Mnist.get_path()
        
        from tools.file_system_tools import FS
        # ignore hidden files/folders (start with a .)
        self.ids = [ each for each in FS.list_folders(self.folder_path) if each[0] != '.' ]
    
    @property
    def path(self):
        return self.folder_path

    @property
    def input_shape(self):
        input_1, _ = self[0]
        return tuple(input_1.shape)

    @property
    def output_shape(self):
        _, output_1 = self[0]
        return tuple(output_1.shape)

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
        from tools.file_system_tools import FS
        from tools.pytorch_tools import to_tensor
        import torch.nn.functional as F

        number = int(FS.read(item_folder+"/data.integer"))
        corrisponding_output = F.one_hot(to_tensor(number), num_classes=self.number_of_classes)
        if self.transform_output:
            corrisponding_output = self.transform_output(corrisponding_output)
        
        # return
        return an_input, corrisponding_output
    
mnist_dataset = Mnist()
