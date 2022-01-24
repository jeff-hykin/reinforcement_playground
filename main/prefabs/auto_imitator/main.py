from super_map import LazyDict, Map

from tools.all_tools import *

from tools.basics import product
from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, misc, Sequential, tensor_to_image, OneHotifier
from tools.schedulers import BasicLearningRateScheduler

class AutoImitator(nn.Module):
    @init.hardware
    def __init__(self, **config):
        super(AutoImitator, self).__init__()
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
        # self.layers.add_module('linear2_activation', nn.Sigmoid())
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1)
        
        # learning rate scheduler
        self.learning_rate_scheduler = BasicLearningRateScheduler(
            value_function=self.learning_rate,
            optimizers=[ self.optimizer ],
        )
        
        self.action_tensor_sum = torch.tensor([0,0,0,0]).float().to(self.hardware)
        self.action_frequency = Map()
        self.one_hotifier = OneHotifier(list(range(self.output_shape[0])))
        # # try to load from path if its given
        # if self.path:
        #     try:
        #         self.load()
        #     except Exception as error:
        #         pass
    
    #
    # manual MSE loss
    #
    @misc.all_args_to_tensor
    @misc.all_args_to_device
    def loss_function(self, model_output_batch, ideal_output_batch):
        which_ideal_actions = ideal_output_batch.long()
        which_ideal_actions_one_hot = to_tensor([ self.one_hotifier.to_one_hot(each) for each in which_ideal_actions ]).to(self.hardware)
        which_model_actions = model_output_batch.argmax(dim=-1)
    
        number_correct_plus_one = 1 + (which_model_actions == which_ideal_actions).sum()/len(which_ideal_actions)
        # ideal output is vector of indicies, model_output_batch is vector of one-hot vectors
        loss = torch.nn.functional.mse_loss(model_output_batch*100, which_ideal_actions_one_hot*100)
        # loss = 10000 * (1/torch.std(model_output_batch, dim=1).sum())
        which_model_actions = which_model_actions.detach()
        for each in model_output_batch:
            self.action_tensor_sum += each
        for each in which_model_actions:
            self.action_frequency[to_pure(each)] += 1
        self.logging.proportion_correct_at_index.append( to_pure(number_correct_plus_one-1) )
        self.logging.loss_at_index.append(to_pure(loss))
        return loss
    
    # #
    # # simple mse
    # #
    # @misc.all_args_to_tensor
    # @misc.all_args_to_device
    # def loss_function(self, model_output_batch, ideal_output_batch):
    #     which_ideal_actions = ideal_output_batch.long()
    #     which_ideal_actions_one_hot = to_tensor([ self.one_hotifier.to_one_hot(each) for each in which_ideal_actions ])
    #     print('model_output_batch.shape = ', model_output_batch.shape)
    #     print('which_ideal_actions_one_hot = ', which_ideal_actions_one_hot)
    #     # ideal output is vector of indicies, model_output_batch is vector of one-hot vectors
    #     loss = torch.nn.functional.mse_loss(model_output_batch, which_ideal_actions_one_hot)
    #     which_model_actions = model_output_batch.detach().argmax(dim=-1)
    #     for each in model_output_batch:
    #         self.action_tensor_sum += each
    #     for each in which_model_actions:
    #         self.action_frequency[to_pure(each)] += 1
    #     self.logging.proportion_correct_at_index.append( (which_model_actions == which_ideal_actions).sum()/len(which_ideal_actions) )
    #     self.logging.loss_at_index.append(to_pure(loss))
    #     return loss
        
    # @misc.all_args_to_tensor
    # @misc.all_args_to_device
    # def loss_function(self, model_output_batch, ideal_output_batch):
    #     which_ideal_actions = ideal_output_batch.long()
    #     # ideal output is vector of indicies, model_output_batch is vector of one-hot vectors
    #     loss = torch.nn.functional.cross_entropy(input=model_output_batch, target=which_ideal_actions)
    #     which_model_actions = model_output_batch.detach().argmax(dim=-1)
    #     for each in model_output_batch:
    #         self.action_tensor_sum += each
    #     for each in which_model_actions:
    #         self.action_frequency[to_pure(each)] += 1
    #     self.logging.proportion_correct_at_index.append( (which_model_actions == which_ideal_actions).sum()/len(which_ideal_actions) )
    #     self.logging.loss_at_index.append(to_pure(loss))
    #     return loss
    
    @misc.all_args_to_tensor
    @misc.all_args_to_device
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        self.learning_rate_scheduler.when_weight_update_starts()
        self.optimizer.zero_grad()
        batch_of_actual_outputs = self.forward(batch_of_inputs)
        loss = self.loss_function(batch_of_actual_outputs, batch_of_ideal_outputs)
        loss.backward()
        self.optimizer.step()
        return loss
    
    @forward.to_tensor
    @forward.to_device
    @forward.to_batched_tensor(number_of_dimensions=4)
    def forward(self, batch_of_inputs):
        # 0 to 1 =>> -1 to 1
        return self.layers.forward(batch_of_inputs)
    
    def save(self, path=None):
        return torch.save(self.state_dict(), path or self.path)

    def load(self, path=None):
        return self.load_state_dict(torch.load(path or self.path))