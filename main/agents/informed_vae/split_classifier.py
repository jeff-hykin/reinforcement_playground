# %%
from tools.all_tools import *

from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network

# Encoder
from agents.informed_vae.encoder import ImageEncoder
# Classifier
from agents.informed_vae.classifier_output import ClassifierOutput
# Decoder
from agents.informed_vae.decoder import ImageDecoder
# %%

class SplitClassifier(nn.Module):
    def __init__(self, **config):
        super(SplitClassifier, self).__init__()
        # 
        # options
        # 
        Network.default_setup(self, config)
        self.input_shape        = config.get('input_shape'       , (1, 28, 28))
        self.latent_shape       = config.get('latent_shape'      , (30,))
        self.output_shape       = config.get('output_shape'      , (2, ))
        self.learning_rate      = config.get('learning_rate'     , 0.01)
        self.momentum           = config.get('momentum'          , 0.5 )
        self.decoder_importance = config.get('decoder_importance', 0.10 )
        model_parameters = ["input_shape", "latent_shape", "output_shape", "learning_rate", "momentum", "decoder_importance"]
        self.record_keeper = self.record_keeper.sub_record_keeper(**{ each: getattr(self, each) for each in model_parameters })
        self.training_record = self.record_keeper.sub_record_keeper(training=True)
        
        # 
        # layers
        # 
        self.add_module('encoder'   , ImageEncoder(input_shape=self.input_shape, output_shape=self.latent_shape))
        self.add_module('classifier', ClassifierOutput(input_shape=self.latent_shape, output_shape=self.output_shape))
        self.add_module('decoder'   , ImageDecoder(input_shape=self.latent_shape, output_shape=self.input_shape))
        
        # 
        # support (optimizer, loss)
        # 
        self.to(self.hardware)
        # create an optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
    @property
    def size_of_last_layer(self):
        return product(self.input_shape if len(self._modules) == 0 else layer_output_shapes(self._modules.values(), self.input_shape)[-1])
    
    def decoder_loss_function(self, model_output, ideal_output):
        return self.decoder_importance * F.mse_loss(model_output.to(self.hardware), ideal_output.to(self.hardware))
    
    def classifier_loss_function(self, model_output, ideal_output):
        # convert from one-hot into number, and send tensor to device
        ideal_output = from_onehot_batch(ideal_output).to(self.hardware)
        return F.nll_loss(model_output, ideal_output)
        
    def forward(self, batch_of_inputs):
        batch_of_inputs          = batch_of_inputs.to(self.hardware)
        batch_of_latent_spaces   = self.encoder.forward(batch_of_inputs)
        batch_of_classifications = self.classifier.forward(batch_of_latent_spaces)
        return batch_of_classifications
    
    def autoencoder_forward(self, batch_of_inputs):
        batch_of_inputs         = batch_of_inputs.to(self.hardware)
        batch_of_latent_spaces  = self.encoder.forward(batch_of_inputs)
        batch_of_decoded_images = self.decoder.forward(batch_of_latent_spaces)
        return batch_of_decoded_images
    
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        self.optimizer.zero_grad()
        record = self.training_record.pending_record
        record["batch_index"] = batch_index
        record["epoch_index"] = epoch_index
        
        latent_space             = self.encoder.forward(batch_of_inputs)
        
        batch_of_classifications = self.classifier.forward(latent_space)
        classifier_loss          = self.classifier_loss_function(batch_of_classifications, batch_of_ideal_outputs)
        classifier_loss.backward(retain_graph=True)
        record["classifier_loss"] = classifier_loss.item()
        
        image_representation = self.decoder.forward(latent_space)
        autoencoder_loss     = self.decoder_loss_function(image_representation, batch_of_inputs)
        autoencoder_loss.backward()
        record["autoencoder_loss"] = autoencoder_loss.item()
        if hasattr(self, "correctness_function") and callable(self.correctness_function):
            record["correct"]  = self.correctness_function(batch_of_classifications, batch_of_ideal_outputs)
            record["total"]    = len(batch_of_classifications)
            record["accuracy"] = round((record["correct"] / record["total"])*100, ndigits = 2)
        
        self.training_record.commit_record()    
        self.optimizer.step()
        return classifier_loss
    
    def fit(self, *, input_output_pairs=None, dataset=None, loader=None, max_epochs=1, batch_size=64, shuffle=True, **kwargs):
        return Network.default_fit(self, input_output_pairs=input_output_pairs, dataset=dataset, loader=loader, max_epochs=max_epochs, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def correctness_function(self, model_batch_output, ideal_batch_output):
        model_batch_output.to(self.hardware)
        ideal_batch_output.to(self.hardware)
        return Network.onehot_correctness_function(self, model_batch_output, ideal_batch_output)

    def test(self, loader, correctness_function=None):
        return Network.default_test(self, loader, loss_function=self.classifier_loss_function)


# %%
# perform test if run directly
# 
if __name__ == "__main__":
    from tools.basics import *
    from tools.ipython_tools import show
    from tools.dataset_tools import binary_mnist
    
    # 
    # perform test on mnist dataset if run directly
    # 
    model = SplitClassifier()
    if not 'train_dataset' in locals(): train_dataset, test_dataset, train_loader, test_loader = quick_loader(binary_mnist([9]), [5, 1])
    model.fit(loader=train_loader, max_epochs=3)
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
        
    # 
    # sample inputs/outputs
    # 
    print("showing samples")
    samples = []
    for each_index in range(15):
        input_data, correct_output = train_dataset[each_index]
        output = model.autoencoder_forward(input_data)
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