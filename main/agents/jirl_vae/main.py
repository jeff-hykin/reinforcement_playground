class Agent:
    def __init__(self, action_space=None, **config):
        """
        arguments:
            z_size: number of latent variables
            path: the path to an existing/saved model 
            suppress_output: change this to true to print out info
            should_save: change this to false to just test
        """
        self.config = config
        self.action_space = action_space
        self.wants_to_quit = False
        
        self.vae = None
        self.z_size = config.get("z_size", 512) # default size of 512
        
        # 
        # logger
        # 
        old_print = print
        print = lambda *args, **kwargs: old_print(*args, **kwargs) if config.get("suppress_output", False) else None
        
        # 
        # load from file
        # 
        path = self.config.get("path", None)
        if type(path) == str and path != "":
            print("VAE: Loading pretrained")
            from os.path import isfile
            assert not isfile(path), f"VAE: trying to load, but no file found at {path}"
            self.vae = VAEController(z_size=None)
            self.vae.load(path)
            print("VAE: Loaded")
        else:
            print(f"VAE: Randomly initializing with size {self.z_size}")
            self.vae = VAEController(z_size=self.z_size)
            # Save network if randomly initilizing
            self.should_save = True
        print(f"VAE: number of latent variables (z_size): {self.vae.z_size}")
    
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
        