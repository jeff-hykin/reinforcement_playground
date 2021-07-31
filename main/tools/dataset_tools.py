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


def binary_mnist(numbers):
    import torchvision
    class Dataset(torchvision.datasets.MNIST):
        number_of_classes = 10
        def __init__(self, *args, **kwargs):
            super(Dataset, self).__init__(*args, **kwargs)
        def __getitem__(self, index):
            an_input, corrisponding_output = super(Dataset, self).__getitem__(index)
            if corrisponding_output in numbers:
                return an_input, torch.tensor([1,0])
            else:
                return an_input, torch.tensor([0,1])
        
    from tools.basics import temp_folder
    options = dict(
        root=f"{temp_folder}/files/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    from torchsampler import ImbalancedDatasetSampler
    
    # 1/6th of the data is for testing
    dataset = Dataset(**options)
    number_of_splits = 6
    test_sections = 1
    number_of_test_elements = int(test_sections * (len(dataset) / 6))
    number_of_train_elements = len(dataset) - number_of_test_elements
    train_dataset, test_dataset = torch.utils.data.random_split(Dataset(**options), [number_of_train_elements, number_of_test_elements])
    # test_dataset = Dataset(**{**options, "train":False})
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=lambda *args:range(len(train_dataset))),
        batch_size=64,
        # shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        # sampler=ImbalancedDatasetSampler(test_dataset),
        batch_size=1000,
        shuffle=True,
    )
    return train_dataset, test_dataset, train_loader, test_loader


