from agents.informed_vae.main import ImageEncoder, ImageModelSequential

import torch
from tools.file_system_tools import FS
from tools.dataset_tools import Mnist
from tools.pytorch_tools import read_image, to_tensor
from torchvision import transforms
import torch.nn.functional as F


# 
# parameters
# 
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

import torch
import torchvision
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


import os
temp_folder_path = f"{os.environ.get('PROJECTR_FOLDER')}/settings/.cache/"
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
# network
#
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(ImageModelSequential):
    def __init__(self):
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
        

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )
            import os
            os.makedirs(f"{temp_folder_path}/results/", exist_ok=True)
            torch.save(network.state_dict(), f"{temp_folder_path}/results/model.pth")
            torch.save(optimizer.state_dict(), f"{temp_folder_path}/results/optimizer.pth")


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
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


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
