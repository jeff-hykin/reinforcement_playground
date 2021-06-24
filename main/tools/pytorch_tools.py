import torch
import torch.nn as nn

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# returns list of tensor sizes
def layer_output_shapes(input_shape, network):
    # convert lists to sequences
    if isinstance(network, list):
        network = nn.Sequential(*network)
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.network = network
        
        def forward(self, x):
            sizes = []
            for layer in self.network:
                x = layer(x)
                sizes.append(x.size())
            return sizes
    
    return Model().forward(torch.ones((1, *input_shape)))