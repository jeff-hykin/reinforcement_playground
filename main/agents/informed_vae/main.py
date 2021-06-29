import torch
import torch.nn as nn
import torch.nn.functional as F

# local 
from tools.defaults import PATHS
from tools.pytorch_tools import ImageModelSequential

# 
# Summary
# 
    # this agent has 3 parts
    # 1. the encoder (from an auto-encoder)
    # 2. the decoder (from an auto-encoder)
    # 3. the "core" agent
    # 
    # the important/novel part is the encoder, the decoder just helps the encoder
    # the "core" agent is arbitrary, however it does need to meet a few requirements
    #     1. needs a value estimation function (estimate the value of a state or state-action pair)
    #     2. a decision gradient (what input affects the action the most)
    #     3. a way to continue backpropogation to update the encoder network
    # 
    # TODO: research prioritized loss functions (get __ correct, if ___ is correct then focus on getting __ correct)
    # 
    # core_agent
    #    the weights are updated based on whatever arbitrary RL method is chosen
    #    however, when the gradient is computed, it (automatically) goes all the way back through the encoder
    #    (it doesn't update the encoder weights but it computes the gradient, as the encoder will use those gradients inside its update function)
    # decoder
    #    the weights are updated in a somewhat strange way, there are two parts that are combined
    #    the first part is simply the squared error loss
    #    the second part is based on either the value estimation function, or decision function of the core_agent
    #        if we encode (compress), decode, and encode-again
    #        then, for the sake of the core_agent, we would want essential features to survive the decoding process
    #        basically decoding should not irreversibly destroy any of the information being heavily utilized by the agent 
    # 
    #        thankfully this isn't too difficult to compute, because we can simply attach a copy of the encoder to 
    #        the end of the decoder to generate a decoded-then-encoded latent space. 
    #        If the core_agent makes the same decision, and has the same value estimation for both 
    #        (e.g. value_of(latent space) ~= value_of(decoded-then-encoded latent space))
    #        then effectively the loss from this is 0, the decoder perfectly preserves information vital to the agent
    #        
    #        by running the forward pass throught the decoder, then encoder, then through the value function of the agent
    #        we can propogate the loss all the way back to the decoder, while keeping the weights of the encoder and 
    #        value function frozen.
    #    these two parts of the loss function (the "image recreation" and "feature preservation") are not equal.
    #    accurately preserving the latent space is the most important, and is always possible to do perfectly
    #    but recreating the image is a useful secondary goal
    #    so we want "feature preservation" to effectively have a higher priority
    #        One such way would be to scale both image-recreation loss and feature-preservation loss between 0 and 1 
    #        and concatonate them with feature-preservation in the 10's place, and image-recreation in the 1's place
    #        (however, this will likely have the tell-tale problems of sigmoid, and might need a different approach)
    # 
    #        an alternative approach would be to have a hyperparameter; a threshold or scaling factor 
    #        so long as the feature-preservation is good enough (within some threshold)
    #        then the loss from the image recreation plays the main role of updating the weights
    # 
    #    NOTE: typical gradient decent may not be the best optimizer here
    #          this is because there should be many different ways to meet the "preserves the latent space" requirement
    #          which might create several local optima
    #          something like the SAM algorithm might help with this: https://github.com/davda54/sam
    #    TODO: consider how this will vary from task to task
    # 
    # encoder
    #    the job of the encoder is 3 ways 
    #    1. dont forget features that were useful for other core_agent's doing other tasks
    #         NOTE: this might need to be done by preserving value-estimation (and/or decision) function for core_agent's on a previous task
    #    2. provide features that are useful to the current core_agent
    #    3. try to help the decompression function with the image-recreation task
    # 

class Agent:
    def __init__(self, action_space=None, **config):
        """
        arguments:
            input_shape: default is 512
            latent_shape: number of latent variables, default is 256
            path: the path to an existing/saved model 
            
            suppress_output: change this to true to print out info
            should_save: change this to false to just test
        """
        self.config = config
        self.action_space = action_space
        self.wants_to_quit = False
        self.print = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
        
        self.input_shape = config.get("input_shape", (3, 32, 32)) # default shape of (3, 32, 32)
        self.latent_shape = config.get("latent_shape", (32)) # default shape of (3, 32, 32)
        self.path = config.get("checkpoint_path", eval(here)+"/previous_checkpoint.pth")
        
        self.observations = []
        self.value_gradients = []
        
        self.encoder = ImageEncoder(
            input_shape=self.input_shape,
            latent_shape=self.latent_shape,
        )
        # instead of the encoder updating weights based only on the decoding process,
        # it will update according to the core_agent, while avoiding overwritting/forgetting old learned weights
        
        # self.decoder = # FIXME
        # self.core_agent = # FIXME
        
        # 
        # load from file
        # 
        path = self.config.get("path", None)
        if type(path) == str and path != "":
            self.print("VAE: Loading pretrained")
            from os.path import isfile
            assert not isfile(path), f"VAE: trying to load, but no file found at {path}"
            self.vae = VAEController(z_size=None)
            self.vae.load(path)
            self.print("VAE: Loaded")
        else:
            self.print(f"VAE: Randomly initializing with size {self.z_size}")
            self.vae = VAEController(z_size=self.z_size)
            # Save network if randomly initilizing
            self.should_save = True
        self.print(f"VAE: number of latent variables (z_size): {self.vae.z_size}")
    
    def on_episode_start(self):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        if type(self.observations) == list:
            # FIXME: train model here
            pass
        
    # this may not be used
    def decide(observation, reward, is_last_timestep):
        """
        returns the action, but in this case the action is a compressed state space
        """
        actual_observation, value_gradient = observation
        self.value_gradients.append(value_gradient)
        return self.model.encode(actual_observation)
    
    
    def on_clean_up(self):
        """
        only called once, and should save checkpoints and cleanup any logging info
        """
        # FIXME: save 
        return
        

# 
# ImageEncoder
# 
class ImageEncoder(ImageModelSequential):
    '''
    examples:
        an_encoder = ImageEncoder()
        from tools.defaults import *
        # img is just a torch tensor
        img = read_image(mnist_dataset.path+"/img_0/data.jpg")
        an_encoder.forward(img)
    notes:
        an external network is going to be the one updating the gradients
        traditionally it would be the decoder, figuring out the best way to decode
        however it can also be the core_agent, figuring out what features would help with its decision process
        Ideally it will be both those things combined, or something more advanced
    '''
    def __init__(self, input_shape=(1, 28, 28), latent_shape=(10,), loss_function=None, **config):
        # this statement is a helper from ImageModelSequential
        with self.setup(input_shape=input_shape, output_shape=latent_shape, loss_function=loss_function):
            # gives us access to
            #     self.print()
            #     self.input_feature_count
            #     self.output_feature_count
            #     self.layers
            #     self.loss()
            #     self.gradients
            #     self.update_gradients()  # using self.loss
            
            # 
            # Layers
            # 
            self.layers.add_module("layer1", nn.Linear(self.size_of_last_layer, int(self.input_feature_count/2)))
            self.layers.add_module("layer1_activation", nn.ReLU())
            self.layers.add_module("layer2", nn.Linear(self.size_of_last_layer, self.output_feature_count))
            self.layers.add_module("layer2_activation", nn.Sigmoid())
            
            # default to squared error loss
            def loss_function(input_batch, ideal_output_batch):
                actual_output_batch = self.forward(input_batch).to(self.device)
                ideal_output_batch = ideal_output_batch.to(self.device)
                return torch.mean((actual_output_batch - ideal_output_batch)**2)
                
            self.loss_function = loss_function
    
    def update_weights(self, input_batch, ideal_outputs_batch, **config):
        # 
        # data used inside the update
        # 
        step_size = config.get("step_size", 0.01)
        gradients = self.compute_gradients_for(
            input_batch=input_batch,
            ideal_outputs_batch=ideal_outputs_batch,
            loss_function=self.loss_function
        )
        # 
        # the actual update
        # 
        # turn off gradient tracking because things are about to be updated
        with torch.no_grad():
            for gradients, each_layer in zip(gradients, self.weighted_layers):
                weight_gradient, bias_gradient = gradients
                
                each_layer.weight += step_size * weight_gradient
                each_layer.bias   += step_size * bias_gradient
                
        # turn gradient-tracking back on
        for each in self.layers:
            each.requires_grad = True
    
    def fit(self, input_output_pairs, **config):
        batch_size     = config.get("batch_size"   , 32)
        epochs         = config.get("epochs"       , 10)
        update_options = config.get("update_options", {}) # step_size can be in here
        
        # convert so that input_batch is a single tensor and ou are a tensor
        all_inputs  = (each for each, _ in input_output_pairs)
        all_outputs = (each for _   , each in input_output_pairs)
        
        from tools.pytorch_tools import batch_input_and_output
        batch_number = 0
        for batch_of_inputs, batch_of_ideal_outputs in batch_input_and_output(all_inputs, all_outputs, batch_size):
            batch_number += 1
            print('batch_number = ', batch_number)
            self.update_weights(batch_of_inputs, batch_of_ideal_outputs, **update_options)
        
        return self

def test_encoder():
    from tools.dataset_tools import mnist_dataset
    from tools.pytorch_tools import read_image
    
    # 
    # forward pass
    # 
    dummy_encoder = ImageEncoder()
    # grab the first Mnist image
    img = read_image(mnist_dataset.path+"/img_0/data.jpg")
    encoded_output = dummy_encoder.forward(img)
    print('encoded_output = ', encoded_output)
    
    # 
    # training
    # 
    return dummy_encoder.fit(
        # mnist_dataset is an iterable with each element being an input output pair
        input_output_pairs=mnist_dataset,
    )


        

        

# # define a simple linear VAE
# class LinearVAE(nn.Module):
#     def __init__(self, latent_shape=None, input_shape=None, **config):
#         super(LinearVAE, self).__init__()
#         self.print = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         self.input_shape = input_shape or (1, 28, 28)
#         self.latent_shape = latent_shape or (16, 1)
        
#         from itertools import product
#         self.input_feature_count = product(self.input_shape)
#         self.encoder_count = 512
#         self.latent_feature_count = product(latent_shape)
        
#         # encoder
#         self.enc1 = nn.Linear(in_features=self.input_feature_count, out_features=self.encoder_count)
#         self.enc2 = nn.Linear(in_features=self.encoder_count, out_features=self.latent_feature_count*2)
 
#         # decoder 
#         self.dec1 = nn.Linear(in_features=self.latent_feature_count, out_features=self.encoder_count)
#         self.dec2 = nn.Linear(in_features=self.encoder_count, out_features=784)

#     def forward(self, input_tensor):
#         latent_tensor = self.encode(input_tensor)
#         reconstruction = self.decode(latent_tensor)
#         return reconstruction, vector_of_means, vector_of_log_variances
    
#     def encode(self, input_tensor):
#         x = self.encode_distribution(input_tensor)
#         # 
#         # reparameterize
#         # 
#         vector_of_means = x[:, 0, :] # the first feature values as mean
#         vector_of_log_variances = x[:, 1, :] # the other feature values as variance
#         standard_deviation = torch.exp(0.5*vector_of_log_variances) # standard deviation
#         vector_of_epsilons = torch.randn_like(standard_deviation) # `randn_like` as we need the same size
#         sample = vector_of_means + (vector_of_epsilons * standard_deviation) # sampling as if coming from the input space
#         latent_tensor = sample
#         return latent_tensor
        
#     def encode_distribution(self, input_tensor):
#         self.print('encode:input_tensor.shape = ', input_tensor.shape)
#         output = self.enc2(
#             F.relu(
#                 self.enc1(input_tensor)
#             )
#         )
#         # reshape into (2, ?) because the first is mean and the second is log of variance
#         return output.view(-1, 2, self.latent_feature_count)
    
#     def decode(self, latent_tensor):
#         self.print('decode:latent_tensor.shape = ', latent_tensor.shape)
#         return torch.sigmoid(
#             self.dec2(
#                 F.relu(
#                     self.dec1(latent_tensor)
#                 )
#             )
#         )
    
#     def run_train(self, train_dataset, validation_dataset, epochs=10, batch_size=64, learning_rate=0.0001):
#         # send to device
#         model = self.to(self.device)
        
#         # 
#         # optimizer
#         # 
#         import torch.optim as optim
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
#         # 
#         # loss
#         # 
#         import torch.nn as nn
#         criterion = nn.BCELoss(reduction='sum') # binary cross entropy loss
#         def final_loss(bce_loss, mu, logvar):
#             """
#             This function will add the reconstruction loss (BCELoss) and the 
#             KL-Divergence.
#             KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#             :param bce_loss: recontruction loss
#             :param mu: the mean from the latent vector
#             :param logvar: log variance from the latent vector
#             """
#             BCE = bce_loss 
#             KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#             return BCE + KLD
        
#         # 
#         # data loader(s)
#         # 
#         from torch.utils.data import DataLoader
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=batch_size,
#             shuffle=True
#         )
#         val_loader = DataLoader(
#             validation_dataset,
#             batch_size=batch_size,
#             shuffle=False
#         )
        
#         # 
#         # train
#         # 
#         def fit(model, dataloader):
#             model.train()
#             running_loss = 0.0
            
#             for index, full_data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
#                 main_data = full_data[0].to(self.device)
#                 reshaped_data = main_data.view(main_data.size(0), -1)
#                 # reset the gradient (because it accumulates) https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
#                 optimizer.zero_grad()
#                 reconstruction, mu, logvar = model(reshaped_data)
#                 bce_loss = criterion(reconstruction, reshaped_data)
#                 loss = final_loss(bce_loss, mu, logvar)
#                 running_loss += loss.item()
#                 loss.backward()
#                 optimizer.step()
                
#             train_loss = running_loss/len(dataloader.dataset)
#             return train_loss
            
        
    
#     def run_train_with_mnist(self, epochs=10, batch_size=64, learning_rate=0.0001):
#         import torchvision.transforms as transforms
#         from torchvision import datasets
#         # transforms
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#         ])
#         train_dataset = datasets.MNIST(
#             root=PATHS["cache_folder"],
#             train=True,
#             download=True,
#             transform=transform,
#         )
#         validation_dataset = datasets.MNIST(
#             root=PATHS["cache_folder"],
#             train=False,
#             download=True,
#             transform=transform,
#         )
#         return run_train(self, train_dataset, validation_dataset, epochs, batch_size, learning_rate)


