# %% 
from tools.all_tools import *

from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network

# Encoder
from agents.informed_vae.encoder import ImageEncoder

# %% 
class EncoderBasedClassifier(nn.Module):
    def __init__(self, **config):
        super(EncoderBasedClassifier, self).__init__()
        # 
        # options
        # 
        Network.default_setup(self, config)
        self.input_shape     = config.get("input_shape"    , (1, 28, 28))
        self.output_shape    = config.get("output_shape"   , (2,))
        self.learning_rate   = config.get("lr"             , 0.01)
        self.momentum        = config.get("momentum"       , 0.5 )
        
        # 
        # layers
        # 
        self.add_module("encoder", ImageEncoder(input_shape=self.input_shape, output_shape=(10,)))
        self.add_module("fc2", nn.Linear(self.size_of_last_layer, product(self.output_shape)))
        self.add_module("fc2_activation", nn.LogSoftmax(dim=1))
        
        # 
        # support (optimizer, loss)
        # 
        self.to(self.device)
        # create an optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
    
    @property
    def size_of_last_layer(self):
        return product(self.input_shape if len(self._modules) == 0 else layer_output_shapes(self._modules.values(), self.input_shape)[-1])
        
    def loss_function(self, model_output, ideal_output):
        # convert from one-hot into number, and send tensor to device
        ideal_output = from_onehot_batch(ideal_output).to(self.device)
        return F.nll_loss(model_output, ideal_output)
    
    def correctness_function(self, model_batch_output, ideal_batch_output):
        return Network.onehot_correctness_function(self, model_batch_output, ideal_batch_output)

    def forward(self, input_data):
        return Network.default_forward(self, input_data)
    
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        return Network.default_update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index)
        
    def fit(self, *, input_output_pairs=None, dataset=None, loader=None, number_of_epochs=3, batch_size=64, shuffle=True):
        return Network.default_fit(self, input_output_pairs=input_output_pairs, dataset=dataset, loader=loader, number_of_epochs=number_of_epochs, batch_size=batch_size, shuffle=shuffle,)
    
    def test(self, loader, correctness_function=None):
        return Network.default_test(self, loader)



if __name__ == "__main__":
    from tools.dataset_tools import binary_mnist
    
    # 
    # perform test on mnist dataset if run directly
    # 
    model = EncoderBasedClassifier()
    train_dataset, test_dataset, train_loader, test_loader = quick_loader(binary_mnist([9]), [5, 1])
    model.fit(loader=train_loader, number_of_epochs=3)
    model.test(loader=test_loader)
    
    # 
    # test inputs/outputs
    # 
    from tools.basics import *
    network = model
    for each_index in range(100):
        input_data, correct_output = train_dataset[each_index]
        # train_dataset, test_dataset, train_loader, test_loader
        guess = [ round(each, ndigits=0) for each in to_pure(network.forward(input_data)) ]
        actual = to_pure(correct_output)
        index = max_index(guess)
        print(f"guess: {guess},\t  index: {index},\t actual: {actual}")

# %%
