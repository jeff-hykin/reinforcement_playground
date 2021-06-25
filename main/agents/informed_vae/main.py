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
        
        self.decoder = # FIXME
        self.core_agent = # FIXME
        
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
    def __init__(self, input_shape=(1, 28, 28), latent_shape=(16, 1), loss=None, **config):
        # 
        # basic setup
        # 
        self.setup(input_shape=input_shape, output_shape=latent_shape, loss=loss)
        # gives us access to
        #     self.print()
        #     self.input_feature_count
        #     self.output_feature_count
        #     self.layers
        #     self.loss()
        #     self.gradients
        #     self.update_gradients()  # using self.loss
        
        # default to squared error loss
        self.loss = loss or (lambda input_batch, ideal_output_batch: torch.mean((self.forward(input_batch) - ideal_output_batch)**2))
        
        # 
        # Layers
        # 
        self.layers.add_module("layer1", nn.Linear(self.size_of_last_layer, int(self.input_feature_count/2)))
        self.layers.add_module("layer1_activation", nn.ReLU())
        self.layers.add_module("layer2", nn.Linear(self.size_of_last_layer, self.output_feature_count))
        self.layers.add_module("layer2_activation", nn.Sigmoid())
    
    def update_weights(self, input_batch, ideal_outputs_batch, step_size=0.01, retain_graph=False):
        self.update_gradients(input_batch, ideal_outputs_batch, retain_graph=retain_graph)
        # turn off tracking because things are about to be updated
        with torch.no_grad():
            for each_layer in self.layers:
                # if it has weights (e.g. not an activation function "layer")
                if hasattr(each_layer, "weight"):
                    each_layer.weight = each_layer.weight - step_size * each_layer.weight.grad
                each_layer.weight.requires_grad = True
                # if it has a bias layer
                if hasattr(each_layer, "bias"):
                    each_layer.bias = each_layer.bias - step_size * each_layer.bias.grad
                each_layer.bias.requires_grad = True

        

# 
# ImageDecoder
# 
class ImageDecoder(nn.Module):
    def __init__(self, latent_shape=None, output_shape=None, loss=None, **config):
        """
        Arguments:
            latent_shape:
                basically the shape of the input
                a tuple, probably with only one large number, e.g. (32, 1) or (32, 1, 1)
                more dynamic shapes are allowed too, e.g (32, 32)
                
            output_shape:
                a tuple that expected to be (image_channels, image_height, image_width)
                where image_channels, image_height, and image_width are all integers
            
            loss:
                a function, that is by default squared error loss
                function Arguments:
                    input_batch:
                        a torch tensor of images with shape (batch_size, channels, height, width)
                    ideal_output_batch:
                        a vector of latent spaces with shape (batch_size, *latent_shape) 
                        for example if the latent_shape was (32, 16) then this would be (batch_size, 32, 16)
                function Content:
                    must perform only pytorch tensor operations on the input_batch
                    (see here for vaild pytorch tensor operations: https://towardsdatascience.com/how-to-train-your-neural-net-tensors-and-autograd-941f2c4cc77c)
                    otherwise pytorch won't be able to keep track of computing the gradient
                function Ouput:
                    should return a torch tensor that is the result of an operation with the input_batch
        """
        # 
        # basic setup
        # 
        super(ImageDecoder, self).__init__()
        self.print = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 
        # model setup
        # 
        self.output_shape = output_shape or (1, 28, 28)
        self.latent_shape = latent_shape or (16, 1)
        # squared error loss
        self.get_loss = loss or lambda input_batch, ideal_output_batch: torch.mean((self.forward(input_batch) - ideal_output_batch)**2)
        from itertools import product
        self.input_feature_count = product(self.output_shape)
        self.latent_feature_count = product(self.latent_shape)
        
        # upgrade image input to 3D if 2D
        if len(output_shape) == 2: output_shape = (1, *output_shape)
        channels, height, width  = output_shape
        
        # 
        # Layers
        # 
        self.layer1 = nn.Linear(self.input_feature_count, self.input_feature_count/2)
        self.layer2 = nn.Linear(self.layer1.out_features, self.latent_feature_count)
        # this var is just so other parts of the code can be automated
        self.layers = [
            self.layer1,
            nn.Relu(),
            self.layer2
            nn.Sigmoid(),
        ]
 
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input_data):
        """
        Arguments:
            input_data:
                either an input image or batch of images
                should be a torch tensor with a shape of (batch_size, channels, height, width)
        Ouptut:
            a torch tensor the shape of the latent space
        Examples:
            obj.forward(torch.tensor([
                # first image in batch
                [
                    # red layer
                    [
                        [ 1, 2, 3 ],
                        [ 4, 5, 6] 
                    ], 
                    # blue layer
                    [
                        [ 1, 2, 3 ],
                        [ 4, 5, 6] 
                    ], 
                    # green layer
                    [
                        [ 1, 2, 3 ],
                        [ 4, 5, 6] 
                    ],
                ] 
            ]))
        
        """
        # converts to torch if needed
        input_data = torch.tensor(input_data)
        # 
        # batch or not?
        # 
        if len(input_data.shape) == 3: 
            batch_size = None
            output_shape = self.latent_shape
            # convert images into batches of 1
            input_data = torch.reshape(input_data, (1, input_data.shape))
        else:
            batch_size = input_data.shape[0]
            output_shape = (batch_size, *self.latent_shape)
            
        neuron_activations = input_data.to(device)
        for each_layer in self.layers:
            neuron_activations = each_layer(neuron_activations)
        
        # force the output to be the correct shape
        return torch.reshape(neuron_activations, output_shape)
    
    @property
    def gradients(self):
        gradients = []
        for each_layer in self.layers:
            # if it has weights (e.g. not an activation function "layer")
            if hasattr(each_layer, "weight"):
                gradients.append(each_layer.weight.grad)
        return gradients
        
    
    def update_gradients(self, input_batch, ideal_outputs_batch, retain_graph=False):
        loss = self.get_loss(input_batch, ideal_outputs_batch)
        # compute the gradients
        loss.backward(retain_graph=retain_graph)
        # return the gradients
        return self.gradients

    def update_weights(self, input_batch, ideal_outputs_batch, step_size=0.01, retain_graph=False):
        self.update_gradients(input_batch, ideal_outputs_batch, retain_graph=retain_graph)
        
        # Note: this is normally where an optimizer would be called
        
        # turn off tracking because things are about to be updated
        with torch.no_grad():
            for each_layer in self.layers:
                # if it has weights (e.g. not an activation function "layer")
                if hasattr(each_layer, "weight"):
                    each_layer.weight = each_layer.weight - step_size * each_layer.weight.grad
                each_layer.weight.requires_grad = True
                
                # if it has a bias layer
                if hasattr(each_layer, "bias"):
                    each_layer.bias = each_layer.bias - step_size * each_layer.bias.grad
                each_layer.bias.requires_grad = True
        

        

# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self, latent_shape=None, input_shape=None, **config):
        super(LinearVAE, self).__init__()
        self.print = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.input_shape = input_shape or (1, 28, 28)
        self.latent_shape = latent_shape or (16, 1)
        
        from itertools import product
        self.input_feature_count = product(self.input_shape)
        self.encoder_count = 512
        self.latent_feature_count = product(latent_shape)
        
        # encoder
        self.enc1 = nn.Linear(in_features=self.input_feature_count, out_features=self.encoder_count)
        self.enc2 = nn.Linear(in_features=self.encoder_count, out_features=self.latent_feature_count*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=self.latent_feature_count, out_features=self.encoder_count)
        self.dec2 = nn.Linear(in_features=self.encoder_count, out_features=784)

    def forward(self, input_tensor):
        latent_tensor = self.encode(input_tensor)
        reconstruction = self.decode(latent_tensor)
        return reconstruction, vector_of_means, vector_of_log_variances
    
    def encode(self, input_tensor):
        x = self.encode_distribution(input_tensor)
        # 
        # reparameterize
        # 
        vector_of_means = x[:, 0, :] # the first feature values as mean
        vector_of_log_variances = x[:, 1, :] # the other feature values as variance
        standard_deviation = torch.exp(0.5*vector_of_log_variances) # standard deviation
        vector_of_epsilons = torch.randn_like(standard_deviation) # `randn_like` as we need the same size
        sample = vector_of_means + (vector_of_epsilons * standard_deviation) # sampling as if coming from the input space
        latent_tensor = sample
        return latent_tensor
        
    def encode_distribution(self, input_tensor):
        self.print('encode:input_tensor.shape = ', input_tensor.shape)
        output = self.enc2(
            F.relu(
                self.enc1(input_tensor)
            )
        )
        # reshape into (2, ?) because the first is mean and the second is log of variance
        return output.view(-1, 2, self.latent_feature_count)
    
    def decode(self, latent_tensor):
        self.print('decode:latent_tensor.shape = ', latent_tensor.shape)
        return torch.sigmoid(
            self.dec2(
                F.relu(
                    self.dec1(latent_tensor)
                )
            )
        )
    
    def run_train(self, train_dataset, validation_dataset, epochs=10, batch_size=64, learning_rate=0.0001):
        # send to device
        model = self.to(self.device)
        
        # 
        # optimizer
        # 
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 
        # loss
        # 
        import torch.nn as nn
        criterion = nn.BCELoss(reduction='sum') # binary cross entropy loss
        def final_loss(bce_loss, mu, logvar):
            """
            This function will add the reconstruction loss (BCELoss) and the 
            KL-Divergence.
            KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            :param bce_loss: recontruction loss
            :param mu: the mean from the latent vector
            :param logvar: log variance from the latent vector
            """
            BCE = bce_loss 
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return BCE + KLD
        
        # 
        # data loader(s)
        # 
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # 
        # train
        # 
        def fit(model, dataloader):
            model.train()
            running_loss = 0.0
            
            for index, full_data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
                main_data = full_data[0].to(self.device)
                reshaped_data = main_data.view(main_data.size(0), -1)
                # reset the gradient (because it accumulates) https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                optimizer.zero_grad()
                reconstruction, mu, logvar = model(reshaped_data)
                bce_loss = criterion(reconstruction, reshaped_data)
                loss = final_loss(bce_loss, mu, logvar)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                
            train_loss = running_loss/len(dataloader.dataset)
            return train_loss
            
        
    
    def run_train_with_mnist(self, epochs=10, batch_size=64, learning_rate=0.0001):
        import torchvision.transforms as transforms
        from torchvision import datasets
        # transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(
            root=PATHS["cache_folder"],
            train=True,
            download=True,
            transform=transform,
        )
        validation_dataset = datasets.MNIST(
            root=PATHS["cache_folder"],
            train=False,
            download=True,
            transform=transform,
        )
        return run_train(self, train_dataset, validation_dataset, epochs, batch_size, learning_rate)
