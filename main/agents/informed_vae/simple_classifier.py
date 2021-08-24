# %% 
from tools.all_tools import *

import torch
from torch.utils.tensorboard import SummaryWriter
from tools.config_tools import PATHS
from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network
from super_map import Map

# Encoder
from agents.informed_vae.encoder import ImageEncoder
# Classifier
from agents.informed_vae.classifier_output import ClassifierOutput
# %% 


class SimpleClassifier(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()
        # 
        # options
        # 
        Network.default_setup(self, config)
        self.input_shape     = config.get("input_shape"    , (1, 28, 28))
        self.mid_shape       = config.get("mid_shape"      , (30,))
        self.output_shape    = config.get("output_shape"   , (2,))
        self.learning_rate   = config.get("learning_rate"  , 0.01)
        self.momentum        = config.get("momentum"       , 0.5 )
        self.fresh           = config.get("fresh"          , False)
        # consider this as a different network, rather than a network setting
        self.record_keeper.parent["fresh"] = self.fresh
        model_parameters = ["input_shape", "mid_shape", "output_shape", "learning_rate", "momentum", ]
        self.record_keeper = self.record_keeper.sub_record_keeper(**{ each: getattr(self, each) for each in model_parameters })
        self.training_record = self.record_keeper.sub_record_keeper(training=True)
        
        # 
        # layers
        # 
        self.add_module("encoder", ImageEncoder(input_shape=self.input_shape, output_shape=self.mid_shape))
        self.add_module("classifier", ClassifierOutput(input_shape=(self.size_of_last_layer,), output_shape=self.output_shape))
        
        
    @property
    def size_of_last_layer(self):
        output = None
        try:
            output = product(self.input_shape if len(tuple(self.children())) == 0 else layer_output_shapes(self.children(), self.input_shape)[-1])
        except Exception as error:
            print("Error getting self.size_of_last_layer", self)
            print('error = ', error)
            raise error
        return output
    
    # [pl.LightningModule]
    def forward(self, input_data):
        return Network.default_forward(self, input_data)
    
    # [pl.LightningModule]
    def training_step(self, batch, batch_index):
        batch_of_inputs, batch_of_ideal_outputs = batch
        record = self.training_record.pending_record
        record["batch_index"] = batch_index
        
        # calculate loss
        batch_of_guesses = self(batch_of_inputs)
        batch_of_ideal_number_outputs = from_onehot_batch(batch_of_ideal_outputs)
        loss = F.nll_loss(batch_of_guesses, batch_of_ideal_number_outputs)
        record["classifier_loss"] = loss.item()
        
        # calculate correctness
        if hasattr(self, "correctness_function") and callable(self.correctness_function):
            record["correct"]  = self.correctness_function(batch_of_guesses     , batch_of_ideal_outputs)
            record["total"]    = len(batch_of_guesses)
            record["accuracy"] = round((record["correct"] / record["total"])*100, ndigits = 2)
        
        self.training_record.commit_record()
        return loss
    
    # [pl.LightningModule]
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return optimizer
    
    @property
    def optimizer(self):
        if not hasattr(self, "_optimizer"): self._optimizer = self.configure_optimizers()
        return self._optimizer
        
    def loss_function(self, model_output, ideal_output):
        # convert from one-hot into number, and send tensor to device
        ideal_output = from_onehot_batch(ideal_output)
        return F.nll_loss(model_output, ideal_output)

    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        return Network.default_update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index)
    
    def fit(self, *, input_output_pairs=None, dataset=None, loader=None, number_of_epochs=3, batch_size=64, shuffle=True, **kwargs):
        return Network.default_fit(self, input_output_pairs=input_output_pairs, dataset=dataset, loader=loader, number_of_epochs=number_of_epochs, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    def correctness_function(self, model_batch_output, ideal_batch_output):
        return Network.onehot_correctness_function(self, model_batch_output, ideal_batch_output)
        
    def test(self, loader, correctness_function=None):
        return Network.default_test(self, loader)

# %% 
if __name__ == "__main__":
    from tools.dataset_tools import binary_mnist
    from pytorch_lightning.loggers import TensorBoardLogger
    
    # 
    # perform test on mnist dataset if run directly
    # 
    model = SimpleClassifier()
    if not 'train_dataset' in locals(): train_dataset, test_dataset, train_loader, test_loader = quick_loader(binary_mnist([9]), [5, 1])
    logger = TensorBoardLogger("lightning_logs", name="simple_classifier")
    
    model.fit(loader=train_loader, max_epochs=3, logger=logger)
    model.test(loader=test_loader)
    
    # 
    # test inputs/outputs
    # 
    from tools.basics import *
    for each_index in range(100):
        input_data, correct_output = train_dataset[each_index]
        # train_dataset, test_dataset, train_loader, test_loader
        guess = [ round(each, ndigits=0) for each in to_pure(model.forward(input_data)) ]
        actual = to_pure(correct_output)
        index = max_index(guess)
        print(f"guess: {guess},\t  index: {index},\t actual: {actual}")

# %%
