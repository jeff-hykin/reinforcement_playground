from tools.all_tools import *

from tools.basics import product
from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, Sequential, tensor_to_image


all_args_to_tensor = forward.all_args_to_tensor
all_args_to_device = forward.all_args_to_device

class AutoImitator(nn.Module):
    @init.hardware
    def __init__(self, **config):
        super(AutoImitator, self).__init__()
        # 
        # options
        # 
        self.input_shape     = config.get('input_shape'    , (1, 28, 28))
        self.latent_shape    = config.get('latent_shape'   , (512,))
        self.output_shape    = config.get('output_shape'   , (10,))
        self.path            = config.get('path'           , None)
        self.learning_rate   = config.get('learning_rate'  , 0.001)
        self.momentum        = config.get('momentum'       , 0.5)
        
        latent_size = product(self.latent_shape)
        # 
        # layers
        # 
        self.layers = Sequential(input_shape=self.input_shape)
        self.encoder = Sequential(input_shape=self.input_shape)
        # 1 input image, 10 output channels, 5x5 square convolution kernel
        self.encoder.add_module('conv1',            nn.Conv2d(self.input_shape[0], 10, kernel_size=5))
        self.encoder.add_module('conv1_pool',       nn.MaxPool2d(2))
        self.encoder.add_module('conv1_activation', nn.ReLU())
        self.encoder.add_module('conv2',            nn.Conv2d(10, 10, kernel_size=5))
        self.encoder.add_module('conv2_drop',       nn.Dropout2d())
        self.encoder.add_module('conv2_pool',       nn.MaxPool2d(2))
        self.encoder.add_module('conv2_activation', nn.ReLU())
        self.encoder.add_module('flatten',          nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
        self.encoder.add_module('fc1',              nn.Linear(self.encoder.output_size, latent_size*2))
        self.encoder.add_module('fc1_activation',   nn.ReLU())
        self.layers.add_module('encoder',          self.encoder)
        self.layers.add_module('fc2',              nn.Linear(self.layers.output_size, latent_size))
        self.layers.add_module('fc2_activation',   nn.ReLU())
        self.layers.add_module('fc3',              nn.Linear(self.layers.output_size, int(latent_size/4)))
        self.layers.add_module('fc3_activation',   nn.ReLU())
        self.layers.add_module('fc4',              nn.Linear(self.layers.output_size, product(self.output_shape)))
        self.layers.add_module('fc4_activation',   nn.Sigmoid())
        
        # optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        # try to load from path if its given
        if self.path:
            try:
                self.load()
            except Exception as error:
                pass
    
    
    @forward.all_args_to_tensor
    @forward.all_args_to_device
    def loss_function(self, model_output, ideal_output):
        # convert to a 0 to 1 range
        model_output = ((model_output+1)/2).squeeze().float()
        ideal_output = ((ideal_output+1)/2).squeeze().float()
        return torch.nn.functional.binary_cross_entropy(model_output, ideal_output)
    
    @forward.all_args_to_tensor
    @forward.all_args_to_device
    def forward(self, batch_of_inputs):
        # 0 to 1 =>> -1 to 1
        return self.layers.forward(batch_of_inputs) * 2 - 1
    
    @all_args_to_tensor
    @all_args_to_device
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        self.optimizer.zero_grad()
        batch_of_actual_outputs = self.forward(batch_of_inputs)
        loss = self.loss_function(batch_of_actual_outputs, batch_of_ideal_outputs)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def save(self, path=None):
        return torch.save(self.state_dict(), path or self.path)

    def load(self, path=None):
        return self.load_state_dict(torch.load(path or self.path))