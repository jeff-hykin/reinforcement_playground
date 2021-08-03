import gdown
import os
import torch
temp_folder_path = f"{os.environ.get('PROJECTR_FOLDER')}/settings/.cache/common_format_datasets"

class BaseDataset(torch.utils.data.Dataset):
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
    
    def __init__(self, transform_input=None, transform_output=None, **kwargs):
        super(torch.utils.data.Dataset, self).__init__(**kwargs)
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

def SimpleDataset(length, getters, data=None, groups=None):
    """
    Arguments:
        getters is a dictionary
        it needs to at least have keys for
            get_input
            get_output
        every value is a function, with two arguments
        as an example
            get_input: lambda self, index: return data[index]
    Example:
        transformed_mnist = torchvision.datasets.MNIST(
            root=f"{temp_folder}/files/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        )
        train, test = SimpleDataset(
            length=len(transformed_mnist),
            groups=[5, 1],
            getters={
                "get_input": lambda self, index: transformed_mnist[index][0],
                "get_output": lambda self, index: transformed_mnist[index][1],
                "get_onehot_output": lambda self, index: onehot_argmax(transformed_mnist[index][1]),
            },
        )
        
    """
    class QuickDataset(torch.utils.data.Dataset):
        def __len__(self):
            return self.length if not callable(length) else length()
        
        def __getitem__(self, index):
            return self.get_input(index), self.get_output(index)
        
        def get_original_index(self, index):
            return index if self.mapping is None else self.mapping[index]
        
        def __init__(self, length, getters, data=None, mapping=None, **kwargs):
            super(QuickDataset).__init__()
            self.length = length
            self.data = data
            self.mapping = mapping
            # create all the getters
            for each_key in getters:
                # exists because of python scoping issues
                def scope_fixer():
                    nonlocal self
                    key_copy = str(each_key)
                    setattr(self, each_key, lambda index, *args, **kwargs: getters[key_copy](self, self.get_original_index(index), *args, **kwargs))
                scope_fixer()
        
    
    main_dataset = QuickDataset(length=length, getters=getters, data=data)
    if groups is None:
        return main_dataset
    else:
        from random import random, sample, choices
        grand_total = len(main_dataset)
        number_of_groups = sum(groups)
        proportions = [ each/number_of_groups for each in groups ]
        total_values = [ int(each * length) for each in proportions ]
        # have the last one be the sum to avoid division/rounding issues
        total_values[-1] = length - sum(total_values[0:-1])
        
        # create a mapping from the new datasets to the original one
        mappings = {}
        indicies_not_yet_used = set(range(grand_total))
        for each_split_index, each_length in enumerate(total_values):
            print('len(indicies_not_yet_used) = ', len(indicies_not_yet_used))
            print('each_length = ', each_length)
            values_for_this_split = set(sample(indicies_not_yet_used, each_length))
            indicies_not_yet_used = indicies_not_yet_used - values_for_this_split
            mappings[each_split_index] = {}
            for each_new_index, each_old_index in enumerate(values_for_this_split):
                mappings[each_split_index][each_new_index] = each_old_index
        
        outputs = []
        for each_split_index, each_length in enumerate(total_values):
            outputs.append(QuickDataset(
                getters=getters,
                data=data,
                length=each_length,
                mapping=mappings[each_split_index],
            ))
        return outputs
        
# FIXME: the images are corrupt for some reason
class Mnist(BaseDataset):
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

    #
    # test/train split
    # 
    # 1/6th of the data is for testing
    dataset = Dataset(**options)
    number_of_splits = 6
    test_sections = 1
    number_of_test_elements = int(test_sections * (len(dataset) / number_of_splits))
    number_of_train_elements = len(dataset) - number_of_test_elements
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,[number_of_train_elements, number_of_test_elements])
    
    # 
    # weighted sampling setup
    # 
    from collections import Counter
    total_number_of_samples = len(train_dataset)
    class_counts = dict(Counter(tuple(each_output.tolist()) for each_input, each_output in train_dataset))
    class_weights = { each_class_key: total_number_of_samples/each_value for each_class_key, each_value in class_counts.items() }
    weights = [ class_weights[tuple(each_output.tolist())] for each_input, each_output in train_dataset ]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), int(total_number_of_samples))
    
    # 
    # create the loaders
    # 
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=64,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
    )
    return train_dataset, test_dataset, train_loader, test_loader

