from super_map import LazyDict, Map

from tools.all_tools import *

from tools.basics import product
from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, misc, Sequential, tensor_to_image, OneHotifier
from tools.schedulers import BasicLearningRateScheduler
from tools.stat_tools import proportionalize, average

class AutoImitatorSingular(nn.Module):
    @init.hardware
    def __init__(self, **config):
        super(AutoImitatorSingular, self).__init__()
        
        # 
        # options
        # 
        self.input_shape     = config.get('input_shape'    , (4, 84, 84))
        self.latent_shape    = config.get('latent_shape'   , (512,))
        self.output_shape    = config.get('output_shape'   , (1,))
        self.path            = config.get('path'           , None)
        self.learning_rate   = config.get('learning_rate'  , 0.00022)
        self.smoothing       = config.get('smoothing'      , 128)
        
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
        self.layers.add_module('linear2_activation', nn.Sigmoid())
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1)
        
        # learning rate scheduler
        self.learning_rate_scheduler = BasicLearningRateScheduler(
            value_function=self.learning_rate,
            optimizers=[ self.optimizer ],
        )
        
        self.prev_output = None
        
        # try to load from path if its given
        if self.path:
            try:
                self.load()
            except Exception as error:
                pass
        
        self.model_total_yes = 0
        self.ideal_total_yes = 0
        self.total_actions = 0
        self.logging = LazyDict(
            proportion_correct_at_index=[],
            loss_at_index=[],
            normal_loss_at_index=[],
        )
    
    @misc.all_args_to_tensor
    @misc.all_args_to_device
    def loss_function(self, model_output_batch, ideal_output_batch):
        ideal_action_yes_no_batch = ideal_output_batch
        model_action_yes_no_batch = model_output_batch.squeeze()
        # update output weights
        self.model_total_yes += model_action_yes_no_batch.sum()
        self.ideal_total_yes += ideal_output_batch.sum()
        self.total_actions += len(ideal_output_batch)
        yes_proportion = (self.ideal_total_yes/self.total_actions)
        weight_for_yes = 1/yes_proportion # only a few yes's => more weight on yes
        weight_for_no  = 1/(1-yes_proportion)
        
        total_loss = torch.tensor(0.0, requires_grad=True)
        for each_model_yes_no, each_ideal_yes_no in zip(model_action_yes_no_batch, ideal_action_yes_no_batch):
            if each_ideal_yes_no:
                total_loss = total_loss + (weight_for_yes * torch.nn.functional.binary_cross_entropy(input=each_model_yes_no, target=each_ideal_yes_no))
            else:
                total_loss = total_loss + (weight_for_no * torch.nn.functional.binary_cross_entropy(input=each_model_yes_no, target=each_ideal_yes_no))
        
        normal_loss = torch.nn.functional.binary_cross_entropy(input=model_action_yes_no_batch, target=ideal_action_yes_no_batch)
        # ideal output is vector of indicies, model_output_batch is vector of one-hot vectors
        model_action_yes_no_batch = model_output_batch.detach().round().long()
        # 
        # logging
        # 
        self.logging.proportion_correct_at_index.append( (model_action_yes_no_batch == ideal_action_yes_no_batch).float().mean() )
        self.logging.loss_at_index.append(total_loss.item())
        self.logging.normal_loss_at_index.append(normal_loss.item())
        return total_loss
    
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
        self.prev_output = self.layers.forward(batch_of_inputs)
        return self.prev_output
    
    def save(self, path=None):
        return torch.save(self.state_dict(), path or self.path)

    def load(self, path=None):
        return self.load_state_dict(torch.load(path or self.path))
    
    def log(self):
        average_loss    = average(self.logging.loss_at_index[-self.smoothing: ])
        average_correct = average(self.logging.proportion_correct_at_index[-self.smoothing: ])
        print(f'''loss: {average_loss:.4f}, correct: {average_correct:.4f}, imitator_action_ratio: {self.logging.get_imitator_action_ratio()}, ideal_action_ratio: {self.logging.get_ideal_action_ratio()}''')

class AutoImitator(nn.Module):
    @init.hardware
    def __init__(self, **config):
        super(AutoImitator, self).__init__()
        
        # 
        # options
        # 
        self.input_shape     = config.get('input_shape'    , (4, 84, 84))
        self.latent_shape    = config.get('latent_shape'   , (512,))
        self.output_shape    = config.get('output_shape'   , (4,))
        self.path            = config.get('path'           , None)
        self.learning_rate   = config.get('learning_rate'  , 0.00022)
        self.smoothing       = config.get('smoothing'      , 128)
        
        self.actions = list(range(product(self.output_shape)))
        self.one_hot_converter = OneHotifier(possible_values=self.actions)
        self.networks = [ AutoImitatorSingular() for each in self.actions ]
        
        self.logging = LazyDict(
            imitator_action_frequency=LazyDict({0:0,1:0,2:0,3:0}),
            ideal_action_frequency=LazyDict({0:0,1:0,2:0,3:0}),
            loss_at_index=[],
            action_count_at_index=[],
            ideal=LazyDict(
                action_0_count_at_index=[],
            ),
            model=LazyDict(
                action_0_count_at_index=[],
            ),
            proportion_correct_at_index=[],
            get_imitator_action_ratio=lambda : "[" + ",".join([ f"{(float(each_value)*100):.2f}%" for each_key, each_value in proportionalize(self.logging.imitator_action_frequency).items() ]) + "]" , 
            get_ideal_action_ratio=   lambda : "[" + ",".join([ f"{(float(each_value)*100):.2f}%" for each_key, each_value in proportionalize(self.logging.ideal_action_frequency   ).items() ]) + "]" , 
        )
    
    @misc.all_args_to_tensor
    @misc.all_args_to_device
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        losses = []
        outputs = []
        correctnesses = []
        output_one_proportion = []
        self.logging.action_count_at_index.append(len(batch_of_ideal_outputs))
        for each_index, each_network in enumerate(self.networks):
            ideal_outputs = (batch_of_ideal_outputs == torch.tensor(each_index)).float()
            
            losses.append(each_network.update_weights(
                batch_of_inputs=batch_of_inputs,
                batch_of_ideal_outputs=ideal_outputs,
                epoch_index=epoch_index,
                batch_index=batch_index,
            ))
            if each_index == 0:
                self.logging.ideal.action_0_count_at_index.append(ideal_outputs.sum())
                self.logging.model.action_0_count_at_index.append(each_network.prev_output.detach().round().sum())
            outputs.append(each_network.prev_output.detach())
            correctnesses.append(each_network.logging.proportion_correct_at_index[-1])
        # this isn't an accurate shape (should basically be transposed), but we pick dim=0 instead of -1 for that reason so it doesn't matter
        model_output = to_tensor(outputs).squeeze().transpose(-1,-2)
        model_action_choices = model_output.argmax(dim=-1)
        self.logging.proportion_correct_at_index.append( (model_action_choices == batch_of_ideal_outputs).float().mean() )
        for each in batch_of_ideal_outputs:
            self.logging.ideal_action_frequency[to_pure(each)] += 1
        for each in model_action_choices:
            self.logging.imitator_action_frequency[to_pure(each.squeeze())] += 1
        self.logging.loss_at_index.append(to_pure(self.loss_function(model_output, batch_of_ideal_outputs)))
        return losses
    
    @misc.all_args_to_tensor
    @misc.all_args_to_device
    def loss_function(self, model_output_batch, ideal_output_batch):
        which_ideal_actions = ideal_output_batch.long()
        # ideal output is vector of indicies, model_output_batch is vector of one-hot vectors
        return torch.nn.functional.cross_entropy(input=model_output_batch, target=which_ideal_actions)
    
    @forward.to_tensor
    @forward.to_device
    @forward.to_batched_tensor(number_of_dimensions=4)
    def forward(self, batch_of_inputs):
        return to_tensor(
            max(each) for each in zip(
                *tuple(each_network.forward(batch_of_inputs) for each_network in self.networks)
            )
        )
    
    def save(self, path=None):
        network_data = tuple(each.state_dict() for each in self.networks)
        path_data = ( self.input_shape, self.latent_shape, self.output_shape, self.path, self.learning_rate, self.smoothing )
        return large_pickle_save((path_data, network_data), path or self.path)

    def load(self, path=None):
        (path_data, network_data) = large_pickle_load(path or self.path)
        ( self.input_shape, self.latent_shape, self.output_shape, self.path, self.learning_rate, self.smoothing ) = path_data
        for each_network, each_state_dict in zip(self.networks, network_data):
            each_network.load_state_dict(each_state_dict)
        return self
    
    def log(self):
        average_losses        = [ f"{average(each.logging.loss_at_index[-self.smoothing: ]):.4f}" for each in self.networks ]
        average_normal_losses = [ f"{average(each.logging.normal_loss_at_index[-self.smoothing: ]):.4f}" for each in self.networks ]
        average_correctnesses = [ f"{average(each.logging.proportion_correct_at_index[-self.smoothing: ]):.4f}" for each in self.networks ]
        average_correct = average(self.logging.proportion_correct_at_index[-self.smoothing: ])
        average_loss    = average(self.logging.loss_at_index[-self.smoothing: ])
        
        ideal0 = f"{self.logging.ideal.action_0_count_at_index[-1]}".rjust(4, " ")
        model0 = f"{self.logging.model.action_0_count_at_index[-1]}".rjust(4, " ")
        print(f'''loss: {average_loss:.4f}, correctness: {average_correct:.3f}, losses: {average_losses}, average_normal_losses: {average_normal_losses}, correctnesses: {average_correctnesses}, ideal0:{ideal0}, model0:{model0}, imitator_action_ratio: {self.logging.get_imitator_action_ratio()}, ideal_action_ratio: {self.logging.get_ideal_action_ratio()}'''.replace('"',''))
    