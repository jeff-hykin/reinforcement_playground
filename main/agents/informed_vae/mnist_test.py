from agents.informed_vae.wip import * 


# 
# auto encoder setup/test
# 
b = SmartImageAutoEncoder()

class AutoMnist(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(AutoMnist, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        an_input, corrisponding_output = super(AutoMnist, self).__getitem__(index)
        return an_input, an_input

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
train_dataset = AutoMnist(**options)
test_dataset = AutoMnist(**{**options, "train":False})
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=True,
)

b.fit(loader=train_loader, number_of_epochs=3)


# 
# importance values
# 
latent_spaces_for_training  = to_tensor( torch.from_numpy(b.encoder(train_dataset[index][0]).cpu().detach().numpy()) for index in range(len(train_dataset)) if index < 10)
latent_spaces_for_testing   = to_tensor( torch.from_numpy(b.encoder(test_dataset[index][0]).cpu().detach().numpy()) for index in range(len(test_dataset)) if index < 1)

import shap

model = nn.Sequential(b.decoder, nn.Flatten())
explainer = shap.DeepExplainer(model, latent_spaces_for_training)
shap_values = explainer.shap_values(latent_spaces_for_testing)


import numpy
import functools
# sum these up elementwise
summed = numpy.squeeze(functools.reduce(
    lambda each_new, existing: numpy.add(each_new, existing),
    # take the absolute value because we just want impactful values, not just neg/pos correlated ones
    numpy.abs(shap_values),
    numpy.zeros_like(shap_values[0]),
))

print('summed = ', summed)