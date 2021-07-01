from agents.informed_vae.main import ImageEncoder, ImageModelSequential

import torch
import torchvision
from tools.file_system_tools import FS
from tools.dataset_tools import Mnist
from tools.pytorch_tools import read_image, to_tensor
from torchvision import transforms
import torch.nn.functional as F

# 
# 
# optimizer
# 
# 
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required

def sgd(
    params,
    d_p_list,
    momentum_buffer_list,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool
):
    r"""
    Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """
    for index, param in enumerate(params):

        d_p = d_p_list[index]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[index]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[index] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            
            sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

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
        
        def NLLLoss(batch_of_actual_outputs, batch_of_ideal_outputs):
            output = batch_of_actual_outputs[range(len(batch_of_ideal_outputs)), batch_of_ideal_outputs]
            return -output.sum()/len(output)
        
        self.loss_function = NLLLoss
        self.optimizer = SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
    
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        self.optimizer.zero_grad()
        batch_of_actual_outputs = self.forward(batch_of_inputs)
        loss = self.loss_function(batch_of_actual_outputs, batch_of_ideal_outputs)
        loss.backward()
        self.optimizer.step()
        return loss
            
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
    batch_size=64,
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
    batch_size=1000,
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
