from agents.informed_vae.main import ImageEncoder, ImageModelSequential

import torch
import torchvision
from tools.file_system_tools import FS
from tools.dataset_tools import Mnist
from tools.pytorch_tools import read_image, to_tensor
from torchvision import transforms
import torch.nn.functional as F
from tools.basics import product

# 
# 
# optimizer
# 
# 
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum subtly differs from
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

    """

    @classmethod
    def sgd(
        cls,
        parameters,
        gradients,
        momentum_buffer_list,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
    ):
        r"""
        Functional API that performs SGD algorithm computation.
        See :class:`~torch.optim.SGD` for details.
        """
        for index, parameters in enumerate(parameters):

            gradient = gradients[index]
            if weight_decay != 0:
                gradient = gradient.add(parameters, alpha=weight_decay)

            if momentum != 0:
                buffer = momentum_buffer_list[index]

                if buffer is None:
                    buffer = torch.clone(gradient).detach()
                    momentum_buffer_list[index] = buffer
                else:
                    buffer.mul_(momentum).add_(gradient, alpha=1 - dampening)

                gradient = buffer

            parameters.add_(gradient, alpha=-lr)
    
    def __init__(self, model_parameters, lr=required, momentum=0, dampening=0, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        super(SGD, self).__init__(model_parameters, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

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
            gradients = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum     = group['momentum']
            dampening    = group['dampening']
            lr           = group['lr']

            for parameter in group['params']:
                if parameter.grad is not None:
                    params_with_grad.append(parameter)
                    gradients.append(parameter.grad)

                    state = self.state[parameter]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            
            self.sgd(
                params_with_grad,
                gradients,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
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
from tools.basics import temp_folder
from tools.pytorch_tools import autoencodeify

class ImageEncoder(ImageModelSequential):
    def __init__(self, **config):
        self.input_shape   = config.get("input_shape", (1, 28, 28))
        self.output_shape  = config.get("output_shape", (10,))
        self.learning_rate = config.get("learning_rate", 0.01)
        self.momentum      = config.get("momentum", 0.5)
        self.log_interval  = config.get("log_interval", 10)
        
        with self.setup(input_shape=self.input_shape, output_shape=self.output_shape):
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
            
            self.layers.add_module("fc2", nn.Linear(self.size_of_last_layer, product(self.output_shape)))
            self.layers.add_module("fc2_activation", nn.LogSoftmax(dim=-1))
        
        def NLLLoss(batch_of_actual_outputs, batch_of_ideal_outputs):
            output = batch_of_actual_outputs[range(len(batch_of_ideal_outputs)), batch_of_ideal_outputs]
            return -output.sum()/len(output)
        
        self.loss_function = NLLLoss
        self.optimizer = SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
    
    def train_and_test_on_mnist(self):
        from tools.basics import temp_folder
        # 
        # training dataset
        # 
        train_loader = torch.utils.data.DataLoader(
            autoencodeify(torchvision.datasets.MNIST)(
                f"{temp_folder}/files/",
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
        # testing dataset
        # 
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                f"{temp_folder}/files/",
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
        
        self.test(test_loader)
        self.fit(loader=train_loader, number_of_epochs=3)
        self.test(test_loader)
        
        return self
        
    
class ImageDecoder(ImageModelSequential):
    def __init__(self, **config):
        self.input_shape   = config.get("input_shape", (10,))
        self.output_shape  = config.get("output_shape", (1, 28, 28))
        self.learning_rate = config.get("learning_rate", 0.01)
        self.momentum      = config.get("momentum", 0.5)
        self.log_interval  = config.get("log_interval", 10)
        
        with self.setup(input_shape=self.input_shape, output_shape=self.output_shape):
            self.layers.add_module("fn1", nn.Linear(self.size_of_last_layer, 400))
            self.layers.add_module("fn1_activation", nn.ReLU(True))
            
            self.layers.add_module("fn2", nn.Linear(self.size_of_last_layer, 4000))
            self.layers.add_module("fn2_activation", nn.ReLU(True))
            
            conv1_shape = [ 10, 20, 20 ] # needs to mupltiply together to be the size of the previous layer (currently 4000)
            conv2_size = 10
            self.layers.add_module("conv1_prep", nn.Unflatten(1, conv1_shape))
            self.layers.add_module("conv1", nn.ConvTranspose2d(conv1_shape[0], conv2_size, kernel_size=5))
            self.layers.add_module("conv2", nn.ConvTranspose2d(conv2_size, 1, kernel_size=5))
            self.layers.add_module("conv2_activation", nn.Sigmoid())
        
            self.loss_function = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
    
class ImageAutoEncoder(ImageModelSequential):
    def __init__(self, **config):
        self.input_shape   = config.get("input_shape", (1, 28, 28))
        self.latent_shape  = config.get("latent_shape", (10,))
        self.output_shape  = config.get("output_shape", (1, 28, 28))
        self.learning_rate = config.get("learning_rate", 0.01)
        self.momentum      = config.get("momentum", 0.5)
        self.log_interval  = config.get("log_interval", 10)
        
        with self.setup(input_shape=self.input_shape, output_shape=self.output_shape):
            # 
            # encoder
            # 
            self.encoder = ImageEncoder(
                input_shape=self.input_shape,
                output_shape=self.latent_shape,
            )
            self.layers.add_module("encoder", self.encoder)
            # 
            # decoder
            # 
            self.decoder = ImageDecoder(
                input_shape=self.latent_shape,
                output_shape=self.output_shape,
            )
            self.layers.add_module("decoder", self.decoder)
            
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
    
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        self.optimizer.zero_grad()
        batch_of_actual_outputs = self.forward(batch_of_inputs)
        loss = self.loss_function(batch_of_actual_outputs, batch_of_inputs)
        loss.backward()
        self.optimizer.step()
        return loss
    
    # TODO: test(self) needs to be changed, but its a bit difficult to make it useful
    
    def train_and_test_on_mnist(self):
        # 
        # modify Mnist so that the input and output are both the image
        # 
        from tools.basics import temp_folder
        train_loader = torch.utils.data.DataLoader(
            autoencodeify(torchvision.datasets.MNIST)(
                f"{temp_folder}/files/",
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
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                f"{temp_folder}/files/",
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
        
        # FIXME: add a testing method (probably related to confusion_matrix) for auto-encoder
        # self.test(test_loader)
        # TODO: autoencodeify the train loader inside the fit function
        self.fit(loader=train_loader, number_of_epochs=3)
        # self.test(test_loader)
        
    
    def generate_confusion_matrix(self, test_loader):
        from tools.basics import product
        number_of_outputs = product(self.latent_shape)
        confusion_matrix = torch.zeros(number_of_outputs, number_of_outputs)
        test_losses = []
        test_loss = 0
        correct = 0
        
        self.eval()
        with torch.no_grad():
            for batch_of_inputs, batch_of_ideal_outputs in test_loader:
                latent_space_activation_batch = self.encoder.forward(batch_of_inputs)
                for each_activation_space, each_ideal_output in zip(latent_space_activation_batch, batch_of_ideal_outputs):
                    # which index was chosen
                    predicted_index = numpy.argmax(each_activation_space)
                    actual_index    = numpy.argmax(each_ideal_output)
                    confusion_matrix[actual_index][predicted_index] += 1
        
        return confusion_matrix
    
    def importance_identification(self, train_dataset, test_dataset, training_size=10, testing_size=1):
        """
        Outputs:
            importance_values, shap_values
            
            the shap_values
                is A list
                The length of list is the size of the model's output (one element per output parameter)
                Every element in the list is a numpy array
                Every element has the shape of the model's input
                A value of 0 means no correlation
                negative values means negatively correlated (similar for positive values)
            
            the importance values
                is a numpy array
                The shape is the same shape as the input for the model
                The values are the relative importance of each latent parameter
                All values are positive
                Larger values = more important
        """
        import shap
        
        # TODO: improve me, these values are converted to and from numpy values basically as a means of copying to shead information such as gradient tracking info
        # note: the inputs are all encoded because we're trying to explain the latent/encoded space
        latent_spaces_for_training  = to_tensor( torch.from_numpy(b.encoder(train_dataset[index][0]).cpu().detach().numpy()) for index in range(len(train_dataset)) if index < training_size)
        latent_spaces_for_testing   = to_tensor( torch.from_numpy(b.encoder(test_dataset[index][0] ).cpu().detach().numpy()) for index in range(len(test_dataset )) if index < testing_size)
        
        # DeepExplainer needs the output to be flat for some reason
        # use only the decoder explains the latent space instead of going back to the image
        model = nn.Sequential(
            self.decoder,
            nn.Flatten(),
        )
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
        return summed, shap_values



class SmartImageAutoEncoder(ImageAutoEncoder):
    def __init__(self, **config):
        self.input_shape   = config.get("input_shape", (1, 28, 28))
        self.latent_shape  = config.get("latent_shape", (20,))
        self.output_shape  = config.get("output_shape", (1, 28, 28))
        self.learning_rate = config.get("learning_rate", 0.01)
        self.momentum      = config.get("momentum", 0.5)
        self.log_interval  = config.get("log_interval", 10)
        
        with self.setup(input_shape=self.input_shape, output_shape=self.output_shape):
            # 
            # encoder
            # 
            self.encoder = ImageEncoder(
                input_shape=self.input_shape,
                output_shape=self.latent_shape,
            )
            self.layers.add_module("encoder", self.encoder)
            # 
            # decoder
            # 
            self.decoder = ImageDecoder(
                input_shape=self.latent_shape,
                output_shape=self.output_shape,
            )
            self.layers.add_module("decoder", self.decoder)
            # 
            # task (classifier)
            # 
            self.task_network = nn.Sequential(
                self.encoder,
                nn.Linear(product(self.latent_shape), 2), # binary classification
            )
            
        self.decoder_loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
    
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        # set the gradient values to zero before accumulating them
        self.zero_grad()
        # loss #1
        batch_of_latent_vectors = self.encoder.forward(batch_of_inputs)
        batch_of_decoded_images = self.decoder.forward(batch_of_latent_vectors)
        decoder_loss = self.decoder_loss_function(batch_of_decoded_images, batch_of_inputs)
        decoder_loss.backward()
        # loss #2 
        batch_of_latent_vectors = self.encoder.forward(batch_of_inputs)
        batch_of_classified_images = self.task_network.forward(batch_of_latent_vectors)
        # FIXME: figure out how to make this loss more important
        task_loss = self.loss_function(batch_of_classified_images, batch_of_inputs)
        task_loss.backward()
        
        self.optimizer.step()
        return loss