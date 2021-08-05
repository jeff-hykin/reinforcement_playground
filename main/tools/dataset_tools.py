#%%
import os
import torch
from tools.basics import *
#%% dataset_tools

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
    class CacheMiss: pass
    def __init__(self, length, getters, attributes=None, data=None, mapping=None, use_cache=False, **kwargs):
        super(QuickDataset).__init__()
        self._cache = {}
        self.length = length
        self.data = data
        self.mapping = mapping
        self.use_cache = use_cache
        attributes = attributes or {}
        self._already_getting_something = False
        self.args = dict(length=length, getters=getters, attributes=attributes, data=data, mapping=mapping, use_cache=use_cache)
        # create all the getters
        for each_key in getters:
            # exists because of python scoping issues
            def scope_fixer():
                nonlocal self
                key_copy = hash(each_key)
                core_getter = getters[each_key]
                def getter(index, *args, **kwargs):
                    # cached values return instantly
                    if self.use_cache:
                        value = self._cache.get((key_copy, index), CacheMiss())
                        if type(value) != CacheMiss:
                            return value
                    
                    # handle recursive case
                    it_was_false = self._already_getting_something == False
                    # use the mapping most of the time
                    if not self._already_getting_something:
                        index =  self.get_original_index(index)
                    # when calling again (recursive) don't re-map the index
                    else:
                        index = index
                    self._already_getting_something = True
                    
                    # actuall run the function
                    output = core_getter(self, index, *args, **kwargs)
                    
                    # cache
                    if self.use_cache: self._cache[key_copy, index] = output
                    # leave it how you found it
                    if it_was_false: self._already_getting_something = False
                    
                    return output
                setattr(self, each_key, getter)
            scope_fixer()
        
        for each_key, each_value in attributes.items():
            setattr(self, each_key, each_value)
    
    def __len__(self):
        return self.length if not callable(self.length) else self.length(self)
    
    def __getitem__(self, index):
        # dont let iterators go out of bounds
        if index >= len(self):
            raise StopIteration
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
        total_values = [ int(each * self.length) for each in proportions ]
        # have the last one be the sum to avoid division/rounding issues
        total_values[-1] = self.length - sum(total_values[0:-1])
        
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
    class_counts = dict(Counter(to_pure(each_output) for each_input, each_output in dataset))
    class_weights = { each_class_key: total_number_of_samples/each_value for each_class_key, each_value in class_counts.items() }
    weights = [ class_weights[to_pure(each_output)] for each_input, each_output in dataset ]
    return torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), int(total_number_of_samples))
        
def quick_mnist(cache=False):
    import torchvision
    original_mnist = torchvision.datasets.MNIST(root=f"{temp_folder}/files/", train=True, download=True,)
    mean, std_deviation = 0.1307, 0.3081
    transformed_mnist = torchvision.datasets.MNIST(
        root=f"{temp_folder}/files/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean,), (std_deviation,)),
            ]
        )
    )
    from tools.pytorch_tools import onehot_argmax, unnormalize
    import torchvision.transforms.functional as TF
    return QuickDataset(
        length=len(original_mnist),
        attributes=dict(
            number_of_classes=10,
            normalizer=torchvision.transforms.Normalize((mean,), (std_deviation,)),
            unnormalizer=lambda image: unnormalize(mean, std_deviation, image),
        ),
        use_cache=cache,
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

def quick_loader(dataset, split, train_batch_size=64, test_batch_size=1000):
    # overwrite the output to be binary classification
    train_dataset, test_dataset = dataset.split(split)
    
    # 
    # create the loaders
    # 
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=create_weighted_sampler_for(train_dataset),
        batch_size=train_batch_size,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=create_weighted_sampler_for(test_dataset),
        batch_size=test_batch_size,
    )
    return train_dataset, test_dataset, train_loader, test_loader

def binary_mnist(numbers):
    return quick_mnist(cache=True).extend(
        get_output= lambda self, index: torch.tensor([1,0]) if self.get_number_value(index) in numbers else torch.tensor([0,1]),
    )

#%%