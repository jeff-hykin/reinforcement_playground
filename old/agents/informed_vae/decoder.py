#%% decoder
from tools.all_tools import *

from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network
#%%

class ImageDecoder(nn.Module):
    def __init__(self, **config):
        super(ImageDecoder, self).__init__()
        # 
        # options
        # 
        Network.default_setup(self, config)
        self.input_shape     = config.get("input_shape"    , (10,))
        channel_count, *image_shape, = self.output_shape = config.get("output_shape"   , (1, 28, 28))
        
        # 
        # layers
        # 
        self.add_module("fn1", nn.Linear(self.size_of_last_layer, 400))
        self.add_module("fn1_activation", nn.ReLU(True))
        conv2_size = 10
        conv2_kernel_size = 5
        conv1_kernel_size = 5
        # element-wise subtraction because kernels add some size
        conv1_image_shape = to_tensor(image_shape) - ((conv2_kernel_size-1) + (conv1_kernel_size-1))
        conv1_image_shape_as_ints_cause_pytorch_is_really_picky = [ int(each) for each in conv1_image_shape ]
        conv1_shape = [ conv2_size*channel_count, *conv1_image_shape_as_ints_cause_pytorch_is_really_picky ]
        self.add_module("fn2", nn.Linear(self.size_of_last_layer, product(conv1_shape)))
        self.add_module("fn2_activation", nn.ReLU(True))
        self.add_module("conv1_prep", nn.Unflatten(1, conv1_shape))
        self.add_module("conv1", nn.ConvTranspose2d(conv1_shape[0], conv2_size, kernel_size=5))
        self.add_module("conv2", nn.ConvTranspose2d(conv2_size, channel_count, kernel_size=5))
        self.add_module("conv2_activation", nn.Sigmoid())
        
        # 
        # support (optimizer, loss)
        # 
        self.to(self.hardware)
        # create an optimizer
        self.loss_function = nn.MSELoss()
    
    @property
    def size_of_last_layer(self):
        return product(self.input_shape if len(self._modules) == 0 else layer_output_shapes(self._modules.values(), self.input_shape)[-1])
        
    def forward(self, input_data):
        return Network.default_forward(self, input_data)
    
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        return Network.default_update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index)
        
    def fit(self, *, input_output_pairs=None, dataset=None, loader=None, max_epochs=1, batch_size=64, shuffle=True):
        return Network.default_fit(self, input_output_pairs=input_output_pairs, dataset=dataset, loader=loader, max_epochs=max_epochs, batch_size=batch_size, shuffle=shuffle,)
    


# %%