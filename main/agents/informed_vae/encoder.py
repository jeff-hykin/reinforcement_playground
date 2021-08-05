#%% encoder
from tools.all_tools import *

from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network
#%%

class ImageEncoder(nn.Module):
    def __init__(self, **config):
        super(ImageEncoder, self).__init__()
        # 
        # options
        # 
        Network.default_setup(self, config)
        self.input_shape     = config.get('input_shape'    , (1, 28, 28))
        self.output_shape    = config.get('output_shape'   , (10,))
        self.batch_size      = config.get('batch_size'     , 64  )
        
        # 
        # layers
        # 
        # 1 input image, 10 output channels, 5x5 square convolution kernel
        self.add_module('conv1', nn.Conv2d(1, 10, kernel_size=5))
        self.add_module('conv1_pool', nn.MaxPool2d(2))
        self.add_module('conv1_activation', nn.ReLU())
        self.add_module('conv2', nn.Conv2d(10, 10, kernel_size=5))
        self.add_module('conv2_drop', nn.Dropout2d())
        self.add_module('conv2_pool', nn.MaxPool2d(2))
        self.add_module('conv2_activation', nn.ReLU())
        self.add_module('flatten', nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
        self.add_module('fc1', nn.Linear(self.size_of_last_layer, product(self.output_shape)))
        self.add_module('fc1_activation', nn.ReLU())
        
        # 
        # support (optimizer, loss)
        # 
        self.to(self.device)
    
    @property
    def size_of_last_layer(self):
        return product(self.input_shape if len(self._modules) == 0 else layer_output_shapes(self._modules.values(), self.input_shape)[-1])
        
    def loss_function(self, model_output, ideal_output):
        # convert from one-hot into number, and send tensor to device
        ideal_output = from_onehot_batch(ideal_output).to(self.device)
        return F.nll_loss(model_output, ideal_output)

    def forward(self, input_data):
        return Network.default_forward(self, input_data)
    
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        return Network.default_update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index)
        
    def fit(self, *, input_output_pairs=None, dataset=None, loader=None, number_of_epochs=3, batch_size=64, shuffle=True):
        return Network.default_fit(self, input_output_pairs=input_output_pairs, dataset=dataset, loader=loader, number_of_epochs=number_of_epochs, batch_size=batch_size, shuffle=shuffle,)

#%%