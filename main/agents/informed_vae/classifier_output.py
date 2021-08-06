# %% 
from tools.all_tools import *

from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network

# %% 
class ClassifierOutput(nn.Module):
    def __init__(self, **config):
        super(ClassifierOutput, self).__init__()
        # 
        # options
        # 
        Network.default_setup(self, config)
        self.input_shape     = config.get("input_shape"    , (1, 28, 28))
        self.output_shape    = config.get("output_shape"   , (2,))
        
        # 
        # layers
        # 
        self.add_module("fc2", nn.Linear(self.size_of_last_layer, product(self.output_shape)))
        self.add_module("fc2_activation", nn.LogSoftmax(dim=1))
        
        # 
        # support (optimizer, loss)
        # 
        self.to(self.hardware)
    
    @property
    def size_of_last_layer(self):
        return product(self.input_shape if len(self._modules) == 0 else layer_output_shapes(self._modules.values(), self.input_shape)[-1])
        
    def forward(self, input_data):
        return Network.default_forward(self, input_data)
    

# %%
