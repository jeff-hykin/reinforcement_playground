from agents.informed_vae.wip import * 

def binary_mnist(numbers):
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
    train_dataset = Dataset(**options)
    test_dataset = Dataset(**{**options, "train":False})
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=ImbalancedDatasetSampler(train_dataset),
        batch_size=64,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        # sampler=ImbalancedDatasetSampler(test_dataset),
        batch_size=1000,
        shuffle=True
    )
    return train_dataset, test_dataset, train_loader, test_loader


# TODO: cross validaion-ish split of data / test sets


# randomize the torch seed
from time import time as now
torch.manual_seed(now())

# 
# 
# 
results = []
for each in [9,3,8]:
    train_dataset, test_dataset, train_loader, test_loader = binary_mnist([each])
    # reset the task network part (last few layers)
    split.task_network = nn.Sequential(
        nn.Linear(product(split.latent_shape), 2), # binary classification
        nn.Sigmoid(),
    )
    split.fit(loader=train_loader, number_of_epochs=3)
    print('#')
    print('# split')
    print('#')
    split_result = split.test(test_loader)
    
    
    classifier.task_network = nn.Sequential(
        nn.Linear(product(classifier.latent_shape), 2), # binary classification
        nn.Sigmoid(),
    )
    classifier.fit(loader=train_loader, number_of_epochs=3)
    print('#')
    print('# classifier')
    print('#')
    classifier_result = classifier.test(test_loader)
    
    new_classifier = ImageClassifier()
    new_classifier.fit(loader=train_loader, number_of_epochs=3)
    print('#')
    print('# new_classifier')
    print('#')
    new_classifier_result = new_classifier.test(test_loader)
    
    results.append({
        "split_train": [],
        "split_test": [],
        "classifier_train": [],
        "classifier_test": [],
        "new_classifier_train": [],
        "new_classifier_test": [],
    })
        [split_result, classifier_result, new_classifier_result])
    
    
# 
# importance values
# 
# latent_spaces_for_training  = to_tensor( torch.from_numpy(b.encoder(train_dataset[index][0]).cpu().detach().numpy()) for index in range(len(train_dataset)) if index < 10)
# latent_spaces_for_testing   = to_tensor( torch.from_numpy(b.encoder(test_dataset[index][0]).cpu().detach().numpy()) for index in range(len(test_dataset)) if index < 1)

# import shap

# model = nn.Sequential(b.decoder, nn.Flatten())
# explainer = shap.DeepExplainer(model, latent_spaces_for_training)
# shap_values = explainer.shap_values(latent_spaces_for_testing)


# import numpy
# import functools
# # sum these up elementwise
# summed = numpy.squeeze(functools.reduce(
#     lambda each_new, existing: numpy.add(each_new, existing),
#     # take the absolute value because we just want impactful values, not just neg/pos correlated ones
#     numpy.abs(shap_values),
#     numpy.zeros_like(shap_values[0]),
# ))

# print('summed = ', summed)