from agents.informed_vae.main import ImageEncoder, ImageModelSequential

import torch
from tools.file_system_tools import FS
from tools.dataset_tools import Mnist
from tools.pytorch_tools import read_image, to_tensor
from torchvision import transforms
import torch.nn.functional as F

import torch
import torchvision
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


#
# network
#
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ImageEncoder(ImageModelSequential):
    def __init__(self, **config):
        self.learning_rate = config.get("learning_rate", 0.01)
        self.momentum      = config.get("momentum", 0.5)
        self.log_interval  = config.get("log_interval", 10)
        
        with self.setup(input_shape=(1, 28, 28), output_shape=(10,)):
            self.layers.add_module("conv1", nn.Conv2d(1, 10, kernel_size=5))
            self.layers.add_module("conv1_pool", nn.MaxPool2d(2))
            self.layers.add_module("conv1_activation", nn.ReLU())
            
            self.layers.add_module("conv2", nn.Conv2d(10, 20, kernel_size=5))
            self.layers.add_module("conv2_dropout", nn.Dropout2d())
            self.layers.add_module("conv2_pool", nn.MaxPool2d(2))
            self.layers.add_module("conv2_activation", nn.ReLU())
            
            self.layers.add_module("flatten", nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
            self.layers.add_module("fc1", nn.Linear(self.size_of_last_layer, 50))
            self.layers.add_module("fc1_activation", nn.ReLU())
            self.layers.add_module("fc1_dropout", nn.Dropout2d())
            
            self.layers.add_module("fc2", nn.Linear(self.size_of_last_layer, 10))
            self.layers.add_module("fc2_activation", nn.LogSoftmax(dim=-1))
        
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
            
    def fit(self, input_output_pairs=None, dataset=None, loader=None, number_of_epochs=3, batch_size=64, shuffle=True):
        """
        Examples:
            model.fit(
                dataset=torchvision.datasets.MNIST(<mnist args>),
                epochs=4,
                batch_size=64,
            )
            
            model.fit(
                loader=torch.utils.data.DataLoader(<dataloader args>),
                epochs=4,
            )
        """
        train_losses = []
        train_counter = []
        if input_output_pairs is not None:
            # creates batches
            def bundle(iterable, bundle_size):
                next_bundle = []
                for each in iterable:
                    next_bundle.append(each)
                    if len(next_bundle) == bundle_size:
                        yield tuple(next_bundle)
                        next_bundle = []
                # return any half-made bundles
                if len(next_bundle) > 0:
                    yield tuple(next_bundle)
            # unpair, batch, then re-pair the inputs and outputs
            input_generator        = (each for each, _ in input_output_pairs)
            ideal_output_generator = (each for _   , each in input_output_pairs)
            seperated_batches = zip(bundle(input_generator, batch_size), bundle(ideal_output_generator, batch_size))
            batches = ((to_tensor(each_input_batch), to_tensor(each_output_batch)) for each_input_batch, each_output_batch in seperated_batches)
                
        # convert the dataset into a loader (assumming loader was not given)
        if isinstance(dataset, torch.utils.data.Dataset):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        if isinstance(loader, torch.utils.data.DataLoader):
            for epoch in range(number_of_epochs):
                epoch += 1
                
                self.train()
                for batch_index, (batch_of_inputs, batch_of_ideal_outputs) in enumerate(loader):
                    self.optimizer.zero_grad()
                    output = self.forward(batch_of_inputs)
                    loss = F.nll_loss(output, batch_of_ideal_outputs)
                    loss.backward()
                    self.optimizer.step()
                    if batch_index % self.log_interval == 0:
                        print(
                            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                                epoch,
                                batch_index * len(batch_of_inputs),
                                len(loader.dataset),
                                100.0 * batch_index / len(loader),
                                loss.item(),
                            )
                        )
                        train_losses.append(loss.item())
                        train_counter.append(
                            (batch_index * 64) + ((epoch - 1) * len(loader.dataset))
                        )
                        # import os
                        # os.makedirs(f"{temp_folder_path}/results/", exist_ok=True)
                        # torch.save(self.state_dict(), f"{temp_folder_path}/results/model.pth")
                        # torch.save(self.optimizer.state_dict(), f"{temp_folder_path}/results/optimizer.pth")
            
    def test(self, test_loader):
        test_losses = []
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_of_inputs, batch_of_ideal_outputs in test_loader:
                actual_output = self(batch_of_inputs)
                test_loss += F.nll_loss(actual_output, batch_of_ideal_outputs, reduction='sum').item()
                prediction = actual_output.data.max(1, keepdim=True)[1]
                correct += prediction.eq(batch_of_ideal_outputs.data.view_as(prediction)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    
# 
# 
# load datasets
# 
# 
batch_size_train = 64
batch_size_test = 1000

import os
temp_folder_path = f"{os.environ.get('PROJECTR_FOLDER')}/settings/.cache/"

# 
# training
# 
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        f"{temp_folder_path}/files/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_train,
    shuffle=True,
)

# 
# testing
# 
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        f"{temp_folder_path}/files/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_test,
    shuffle=True,
)


# 
# 
# train and test the model
# 
# 
network = ImageEncoder()
network.test(test_loader)
network.fit(loader=train_loader, number_of_epochs=3)
network.test(test_loader)
