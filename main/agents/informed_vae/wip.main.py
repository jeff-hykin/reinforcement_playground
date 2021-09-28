import torch
import torch.nn as nn
import torch.nn.functional as F

# local 
from tools.all_tools import PATHS
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
        self.wants_to_end_episode = False
        self.show = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
        
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
            self.show("VAE: Loading pretrained")
            from os.path import isfile
            assert not isfile(path), f"VAE: trying to load, but no file found at {path}"
            self.vae = VAEController(z_size=None)
            self.vae.load(path)
            self.show("VAE: Loaded")
        else:
            self.show(f"VAE: Randomly initializing with size {self.z_size}")
            self.vae = VAEController(z_size=self.z_size)
            # Save network if randomly initilizing
            self.should_save = True
        self.show(f"VAE: number of latent variables (z_size): {self.vae.z_size}")
    
    def when_episode_starts(self, initial_observation, episode_index):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        if type(self.observations) == list:
            # FIXME: train model here
            pass
        
    def when_action_needed(self, observation, reward):
        """
        returns the action, but in this case the action is a compressed state space
        """
        actual_observation, value_gradient = observation
        self.value_gradients.append(value_gradient)
        return self.model.encode(actual_observation)
    
    
    def when_should_clean(self):
        """
        only called once, and should save checkpoints and cleanup any logging info
        """
        # FIXME: save 
        return
