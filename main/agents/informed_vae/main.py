import torch
import torch.nn as nn
import torch.nn.functional as F

# local 
from tools.defaults import PATHS

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
        
        self.model = None
        
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
    
    # this may not be used
    def decide(observation, reward, is_last_timestep):
        """
        returns (observation, reward, is_last_timestep)
        but the observation is compressed
        """
        compressed_observation = self.enhace_observation(observation)
        return compressed_observation
    
    def enhace_observation(observation):
        return self.vae.encode_from_raw_image(observation)
    
    def on_episode_start(self):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        return
    
    def on_clean_up(self):
        """
        only called once, and should save checkpoints and cleanup any logging info
        """
        return
        




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
