from super_map import LazyDict

from tools.all_tools import *

from tools.basics import product
from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, Sequential, tensor_to_image
from tools.schedulers import BasicLearningRateScheduler

class AutoImitator(nn.Module):
    def __init__(self, **config):
        super(AutoImitator, self).__init__()
        # put in a method for load/save reaons
        self.construct(**config)
    
    @init.hardware
    def construct(self, **config):
        self.config = config
        self.logging = LazyDict(proportion_correct_at_index=[], loss_at_index=[])
        # 
        # options
        # 
        self.input_shape     = config.get('input_shape'    , (4, 84, 84))
        self.latent_shape    = config.get('latent_shape'   , (512,))
        self.output_shape    = config.get('output_shape'   , (4,))
        self.path            = config.get('path'           , None)
        self.learning_rate   = config.get('learning_rate'  , 0.001)
        
        latent_size = product(self.latent_shape)
        # 
        # layers
        # 
        self.layers = Sequential(input_shape=self.input_shape)
        self.encoder = Sequential(input_shape=self.input_shape)
        
        self.encoder.add_module('conv1'           , nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4, padding=0))
        self.encoder.add_module('conv1_activation', nn.ReLU())
        self.encoder.add_module('conv2'           , nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0))
        self.encoder.add_module('conv2_activation', nn.ReLU())
        self.encoder.add_module('conv3'           , nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))
        self.encoder.add_module('conv3_activation', nn.ReLU())
        self.encoder.add_module('flatten'         , nn.Flatten(start_dim=1, end_dim=-1)) # 1 => skip the first dimension because thats the batch dimension
        self.encoder.add_module('linear1'         , nn.Linear(in_features=self.encoder.output_size, out_features=latent_size, bias=True)) 
        
        self.layers.add_module('encoder', self.encoder)
        self.layers.add_module('linear2'           , nn.Linear(in_features=latent_size, out_features=product(self.output_shape), bias=True),)
        self.layers.add_module('linear2_activation', nn.Softmax(dim=0))
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1)
        
        # learning rate scheduler
        self.learning_rate_scheduler = BasicLearningRateScheduler(
            value_function=self.learning_rate,
            optimizers=[ self.optimizer ],
        )
        
        # try to load from path if its given
        if self.path:
            try:
                self.load()
            except Exception as error:
                pass
    
    @forward.all_args_to_tensor
    @forward.all_args_to_device
    def loss_function(self, model_output, ideal_output):
        which_ideal_actions = ideal_output.long()
        # ideal output is vector of indicies, model_output is vector of one-hot vectors
        loss = torch.nn.functional.cross_entropy(input=model_output, target=which_ideal_actions)
        which_model_actions = model_output.detach().argmax(dim=-1)
        self.logging.proportion_correct_at_index.append( (which_model_actions == which_ideal_actions).sum()/len(which_ideal_actions) )
        self.logging.loss_at_index.append(to_pure(loss))
        return loss
    
    @forward.all_args_to_tensor
    @forward.all_args_to_device
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        self.learning_rate_scheduler.when_weight_update_starts()
        self.optimizer.zero_grad()
        batch_of_actual_outputs = self.forward(batch_of_inputs)
        loss = self.loss_function(batch_of_actual_outputs, batch_of_ideal_outputs)
        loss.backward()
        self.optimizer.step()
        return loss
    
    @forward.all_args_to_tensor
    @forward.all_args_to_device
    def forward(self, batch_of_inputs):
        # 0 to 1 =>> -1 to 1
        return self.layers.forward(batch_of_inputs)
    
    def save(self, path=None):
        data = (
            self.state_dict() ,
            self.config
        )
        large_pickle_save(data, file_path=(path or self.path))
        return self

    def load(self, path=None):
        (
            state_dict,
            self.config,
        ) = large_pickle_load(path or self.path)
        # set everything up
        self.construct(**self.config)
        # then load weights back in
        self.load_state_dict(state_dict)
        return self
    
    def save(self, path=None):
        data = (
            self.state_dict() ,
            self.config
        )
        large_pickle_save(data, file_path=(path or self.path))
        return self

    def load(self, path=None):
        (
            state_dict,
            self.config,
        ) = large_pickle_load(path or self.path)
        # set everything up
        self.construct(**self.config)
        # then load weights back in
        self.load_state_dict(state_dict)
        return self