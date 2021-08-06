# %%
from tools.all_tools import *

from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network

# Encoder
from agents.informed_vae.encoder import ImageEncoder
# Decoder
from agents.informed_vae.decoder import ImageDecoder
# %%

class ImageAutoEncoder(nn.Module):
    def __init__(self, **config):
        super(ImageAutoEncoder, self).__init__()
        # 
        # options
        # 
        Network.default_setup(self, config)
        self.input_shape     = config.get('input_shape'    , (1, 28, 28))
        self.latent_shape    = config.get('latent_shape'   , (30,))
        self.output_shape    = config.get('output_shape'   , (1, 28, 28))
        self.learning_rate   = config.get('lr'             , 0.01)
        self.momentum        = config.get('momentum'       , 0.5 )
        
        # 
        # layers
        # 
        self.add_module('encoder', ImageEncoder(input_shape=self.input_shape, output_shape=self.latent_shape))
        self.add_module('decoder', ImageDecoder(input_shape=self.latent_shape, output_shape=self.output_shape))
        
        # 
        # support (optimizer, loss)
        # 
        self.to(self.hardware)
        # create an optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        # use mean squared error
        self.loss_function = nn.MSELoss()
    
    @property
    def size_of_last_layer(self):
        return product(self.input_shape if len(self._modules) == 0 else layer_output_shapes(self._modules.values(), self.input_shape)[-1])
    
    def loss_function(self, model_output, ideal_output):
        return F.mse_loss(model_output.to(self.hardware), ideal_output.to(self.hardware))
        
    def forward(self, input_data):
        input_data.to(self.hardware)
        latent_space = self.encoder.forward(input_data)
        output = self.decoder.forward(latent_space)
        return output
        # return Network.default_forward(self, input_data)
    
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        return Network.default_update_weights(self, batch_of_inputs, batch_of_inputs, epoch_index, batch_index)
        
    def fit(self, *, input_output_pairs=None, dataset=None, loader=None, number_of_epochs=3, batch_size=64, shuffle=True):
        return Network.default_fit(self, input_output_pairs=input_output_pairs, dataset=dataset, loader=loader, number_of_epochs=number_of_epochs, batch_size=batch_size, shuffle=shuffle,)

# %%
# perform test if run directly
# 
if __name__ == '__main__':
    from tools.dataset_tools import binary_mnist
    from tools.basics import *
    from tools.ipython_tools import show
    
    model = ImageAutoEncoder()
    try:
        train_dataset[0]
    except Exception as error:
        # doesn't matter that its binary mnist cause the autoencoder only uses input anyways
        train_dataset, test_dataset, train_loader, test_loader = quick_loader(binary_mnist([9]), [5, 1])
    
    model.fit(loader=train_loader, number_of_epochs=3)
        
    # 
    # sample inputs/outputs
    # 
    print("showing samples")
    samples = []
    for each_index in range(100):
        input_data, correct_output = train_dataset[each_index]
        output = model.forward(input_data)
        sample =  torch.cat(
            (
                train_dataset.unnormalizer(input_data.to(model.device)),
                train_dataset.unnormalizer(output.to(model.device)),
            ), 
            1
        )
        show(image_tensor=sample)
        samples.append(sample)

# %%
