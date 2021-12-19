from tools.all_tools import *

from torchvision import datasets, transforms
from tools.basics import product, flatten
from tools.pytorch_tools import Network

def create_network():
    input_shape = env.observation_space.shape

class A2cNetwork(nn.Module):
    def __init__(self, **config):
        super(A2cNetwork, self).__init__()
        # 
        # options
        # 
        Network.default_setup(self, config)
        self.input_shape        = config.get('input_shape'       , (1, 28, 28))
        self.layer_sizes        = config.get('layer_sizes'       , [64,64])
        self.actor_output_size  = config.get('actor_output_size' , (2,))
        self.learning_rate      = config.get('learning_rate'     , 0.01)
        self.momentum           = config.get('momentum'          , 0.5 )
        
        # 
        # common layers
        # 
        last_common_layer_size = self.layer_sizes.pop()
        self.layers = nn.Sequential(list(flatten(
            (
                nn.Linear(self.size_of_last_layer, each_size),
                nn.ReLU()
            ) for each_size in self.layer_sizes
        )))
        size_before_last_layer = self.size_of_last_layer
        
        # 
        # actor output
        # 
        self.actor_layers = nn.Sequential([
            nn.Linear(size_before_last_layer, last_common_layer_size),
            nn.Tanh(),
            nn.Linear(last_common_layer_size, self.actor_output_size),
            nn.Tanh(),
            nn.Softmax(),
        ])
        
        # 
        # Critic output
        # 
        self.critic_layers = nn.Sequential([
            nn.Linear(size_before_last_layer, last_common_layer_size),
            nn.ReLU(),
            nn.Linear(last_common_layer_size, 1),
        ])
        
        # housekeeping
        self.to(self.device)
        
        # 
        # optimizer
        # 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # 
        # loss
        # 
        # args:
        #     advantage: a matrix, each row is an action that is one-hot encoded
        #     predicted_output: vector of probabilies (one for each acton)
        # returns:
        #     integral of the policy gradient, a one-hot encoded value for each action a_t
        self.actor_loss = lambda advantage, predicted_output: torch.sum(advantage * -torch.log(torch.clip(predicted_output, 1e-8, 1-1e-8)))
        self.critic_loss = lambda advantage, predicted_output: torch.sum(-advantage * predicted_output)
        
    
    @property
    def size_of_last_layer(self):
        return product(self.input_shape if len(self.layers) == 0 else layer_output_shapes(self.input_shape, self.layers)[-1])
        
    def loss_function(self, model_output, ideal_output):
        # convert from one-hot into number, and send tensor to device
        ideal_output = from_onehot_batch(ideal_output).to(self.device)
        return F.nll_loss(model_output, ideal_output)
    
    def correctness_function(self, model_batch_output, ideal_batch_output):
        return Network.onehot_correctness_function(self, model_batch_output, ideal_batch_output)

    def forward(self, input_data):
        return Network.default_forward(self, input_data)
    
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        return Network.default_update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index)
        
    def fit(self, *, input_output_pairs=None, dataset=None, loader=None, number_of_epochs=3, batch_size=64, shuffle=True):
        return Network.default_fit(self, input_output_pairs=input_output_pairs, dataset=dataset, loader=loader, number_of_epochs=number_of_epochs, batch_size=batch_size, shuffle=shuffle,)
    
    def test(self, loader, correctness_function=None):
        return Network.default_test(self, loader)

# 
# perform test if run directly
# 
if __name__ == '__main__':
    from tools.dataset_tools import *
    from tools.basics import *
    
    model = A2cNetwork()
    # FIXME: dataset
    train_dataset, test_dataset, train_loader, test_loader = binary_mnist([9])
    model.fit(loader=train_loader, number_of_epochs=3)
    model.test(loader=test_loader)
    
    # 
    # test inputs/outputs
    # 
    for each_index in range(100):
        input_data, correct_output = train_dataset[each_index]
        # train_dataset, test_dataset, train_loader, test_loader
        guess = [ round(each, ndigits=0) for each in to_pure(model.forward(input_data)) ]
        actual = to_pure(correct_output)
        index = max_index(guess)
        print(f'guess: {guess},	  index: {index},	 actual: {actual}')