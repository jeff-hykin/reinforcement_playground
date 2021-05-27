import torch
import torchvision

# local 
from tools.defaults import *

class Agent:
    def __init__(self, action_space=None, categories, **config):
        """
        arguments:
            categories: the dataset.categories
            checkpoint_path: path to an existing model
            suppress_output: set to true to disable printing
        """
        self.config = config
        self.action_space = action_space
        self.wants_to_quit = False
        #
        # logger
        # 
        self.print = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
        
        # based on https://github.com/NVIDIA-AI-IOT/jetracer/blob/master/notebooks/interactive_regression.ipynb
        
        self.path = config.get("checkpoint_path", eval(here)+"/previous_checkpoint.pth")
        self.optimizer = torch.optim.Adam(model.parameters())
        self.output_shape = 2 * len(categories)  # x, y coordinate for each category
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(512, self.output_shape)
        # use cuda if possible
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.model = model.to(device)
            self.print('agent is using cuda')
        else:
            self.print('cuda doesnt seem to be available')
    
    # this may not be used
    def decide(observation, reward, is_last_timestep):
        """
        returns the action
        """
        return self.model.eval(observation)
    
    
    def on_episode_start(self):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        self.save_model()
    
    def on_clean_up(self):
        """
        only called once, and should save checkpoints and cleanup any logging info
        """
        return
        
    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        
    def save_model(self):
        self.print("saving model")
        torch.save(self.model.state_dict(), self.checkpoint_path)
        
    def train(self):
        print("not yet implemented")

    def test(self):
        print("not yet implemented")