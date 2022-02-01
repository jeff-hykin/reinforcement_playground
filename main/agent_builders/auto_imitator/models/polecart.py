from super_map import LazyDict, Map

from tools.all_tools import *

from tools.basics import product
from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, misc, Sequential, tensor_to_image, OneHotifier
from tools.schedulers import BasicLearningRateScheduler
from tools.stat_tools import proportionalize, average

class AutoImitator(nn.Module):
    @init.hardware
    def __init__(self, **config):
        super(AutoImitator, self).__init__()
        
        # 
        # options
        # 
        self.input_shape     = config.get('input_shape'    , (4, ))
        self.latent_shape    = config.get('latent_shape'   , (512,))
        self.output_shape    = config.get('output_shape'   , (2,))
        self.path            = config.get('path'           , None)
        self.learning_rate   = config.get('learning_rate'  , 0.001)
        self.smoothing       = config.get('smoothing'      , 128)
        
        latent_size = product(self.latent_shape)
        # 
        # layers
        # 
        self.layers = Sequential(input_shape=self.input_shape)
        
        self.layers.add_module('l1', nn.Linear(product(self.input_shape), 64))
        self.layers.add_module('l2', nn.Tanh())
        self.layers.add_module('l3', nn.Linear(64, 32))
        self.layers.add_module('l4', nn.Tanh())
        self.layers.add_module('l5', nn.Linear(32, product(self.output_shape)))
        self.layers.add_module('l6', nn.Softmax(dim=0))
        
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
        
        self.logging = LazyDict(
            imitator_action_frequency=LazyDict({0:0,1:0,2:0,3:0}),
            ideal_action_frequency=LazyDict({0:0,1:0,2:0,3:0}),
            proportion_correct_at_index=[],
            loss_at_index=[],
            get_imitator_action_ratio=lambda : "[" + ",".join([ f"{(float(each_value)*100):.2f}%" for each_key, each_value in proportionalize(self.logging.imitator_action_frequency).items() ]) + "]" , 
            get_ideal_action_ratio=   lambda : "[" + ",".join([ f"{(float(each_value)*100):.2f}%" for each_key, each_value in proportionalize(self.logging.ideal_action_frequency   ).items() ]) + "]" , 
        )
    
    @misc.all_args_to_tensor
    @misc.all_args_to_device
    def loss_function(self, model_output_batch, ideal_output_batch):
        loss = torch.nn.functional.cross_entropy(input=model_output_batch, target=ideal_output_batch.long())
        self.logging.loss_at_index.append(to_pure(loss))
        return loss
    
    @misc.all_args_to_tensor
    @misc.all_args_to_device
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        self.learning_rate_scheduler.when_weight_update_starts()
        self.optimizer.zero_grad()
        batch_of_actual_outputs = self.forward(batch_of_inputs)
        loss = self.loss_function(batch_of_actual_outputs, batch_of_ideal_outputs)
        print(f'''loss = {loss}''')
        loss.backward()
        self.optimizer.step()
        return loss
    
    @forward.to_tensor
    @forward.to_device
    @forward.to_batched_tensor(number_of_dimensions=2)
    def forward(self, batch_of_inputs):
        probs = self.layers.forward(batch_of_inputs).float()
        return probs
    
    def save(self, path=None):
        return torch.save(self.state_dict(), path or self.path)

    def load(self, path=None):
        return self.load_state_dict(torch.load(path or self.path))
    
    def log(self):
        average_loss    = average(self.logging.loss_at_index[-self.smoothing: ])
        average_correct = average(self.logging.proportion_correct_at_index[-self.smoothing: ])
        print(f'''loss: {average_loss:.4f}, correct: {average_correct:.4f}, imitator_action_ratio: {self.logging.get_imitator_action_ratio()}, ideal_action_ratio: {self.logging.get_ideal_action_ratio()}''')
    