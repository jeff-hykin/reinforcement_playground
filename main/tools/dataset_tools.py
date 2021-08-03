import os
import torch
temp_folder_path = f"{os.environ.get('PROJECTR_FOLDER')}/settings/.cache/common_format_datasets"

class QuickDataset(torch.utils.data.Dataset):
    """
    Arguments:
        length: (required)
            length of the dataset
        getters: (required)
            a dictionary
            it needs to at least have keys for
                get_input
                get_output
            every value is a function, with two arguments
            as an example
                get_input: lambda self, index: return data[index]
        attributes: (optional)
            a dictionary of values that will be attached to 
            the dataset, and all splits of the dataset
    Example:
        mnist = torchvision.datasets.MNIST(*args)
        train, test = QuickDataset(
            length=len(mnist),
            getters=dict(
                get_input= lambda self, index: mnist[index][0],
                get_output= lambda self, index: mnist[index][1],
                get_onehot_output= lambda self, index: onehotify(mnist[index][1]),
            ),
        ).split([5,1]) # ratio of 5 to 1 for train, test
        
    """
    def __init__(self, length, getters, attributes=None, data=None, mapping=None, **kwargs):
        super(QuickDataset).__init__()
        self.length = length
        self.data = data
        self.mapping = mapping
        attributes = attributes or {}
        self.args = dict(length=length, getters=getters, attributes=attributes, data=data, mapping=mapping)
        # create all the getters
        for each_key in getters:
            # exists because of python scoping issues
            def scope_fixer():
                nonlocal self
                key_copy = str(each_key)
                setattr(self, each_key, lambda index, *args, **kwargs: getters[key_copy](self, self.get_original_index(index), *args, **kwargs))
            scope_fixer()
        
        for each_key, each_value in attributes.items():
            setattr(self, each_key, each_value)
    
    def __len__(self):
        return self.length if not callable(length) else length()
    
    def __getitem__(self, index):
        return self.get_input(index), self.get_output(index)
    
    def get_original_index(self, index):
        return index if self.mapping is None else self.mapping[index]
    
    def extend(self, **new_getters):
        return QuickDataset(**{
            **self.args,
            "getters": {
                **self.args["getters"],
                **new_getters
            }
        })
    
    def split(self, groups):
        from random import random, sample, choices
        grand_total = len(self)
        number_of_groups = sum(groups)
        proportions = [ each/number_of_groups for each in groups ]
        total_values = [ int(each * length) for each in proportions ]
        # have the last one be the sum to avoid division/rounding issues
        total_values[-1] = length - sum(total_values[0:-1])
        
        # create a mapping from the new datasets to the original one
        mappings = {}
        indicies_not_yet_used = set(range(grand_total))
        for each_split_index, each_length in enumerate(total_values):
            values_for_this_split = set(sample(indicies_not_yet_used, each_length))
            indicies_not_yet_used = indicies_not_yet_used - values_for_this_split
            mappings[each_split_index] = {}
            for each_new_index, each_old_index in enumerate(values_for_this_split):
                mappings[each_split_index][each_new_index] = each_old_index
        
        outputs = []
        for each_split_index, each_length in enumerate(total_values):
            outputs.append(QuickDataset(
                getters=self.args["getters"],
                attributes=self.args["attributes"],
                data=self.args["data"],
                length=each_length,
                mapping=mappings[each_split_index],
            ))
        return outputs

def create_weighted_sampler_for(dataset):
    from collections import Counter
    total_number_of_samples = len(dataset)
    class_counts = dict(Counter(tuple(each_output.tolist()) for each_input, each_output in dataset))
    class_weights = { each_class_key: total_number_of_samples/each_value for each_class_key, each_value in class_counts.items() }
    weights = [ class_weights[tuple(each_output.tolist())] for each_input, each_output in dataset ]
    return torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), int(total_number_of_samples))
        
def quick_mnist():
    original_mnist = torchvision.datasets.MNIST(root=f"{temp_folder}/files/", train=True, download=True,)
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
    from tools.pytorch_tools import onehot_argmax
    import torchvision.transforms.functional as TF
    return QuickDataset(
        length=len(original_mnist),
        attributes=dict(
            number_of_classes=10
        ),
        getters=dict(
            # normalized
            get_input= lambda self, index: transformed_mnist[index][0],
            # onehot-ified
            get_output= lambda self, index: onehot_argmax(transformed_mnist[index][1]),
            # misc
            get_image= lambda self, index: original_mnist[index][0],
            get_image_tensor= lambda self, index: TF.to_tensor(original_mnist[index][0]),
            get_number_value= lambda self, index: original_mnist[index][1],
        ),
    )

def binary_mnist(numbers):
    # overwrite the output to be binary classification
    binary_dataset = quick_mnist().extend(
        get_output= lambda self, index: 1 if self.get_number_value(index) in numbers else 0,
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

