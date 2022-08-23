import math

import torch
import torch.nn as nn
from trivial_torch_tools import Sequential, init, convert_each_arg, to_tensor
from trivial_torch_tools.generics import product, to_pure
from super_map import LazyDict

mse_loss = nn.MSELoss()
class GeneralApproximator(nn.Module): # NOTE: this thing is broken! it should work but for some reason fails to learn abstractions
    @init.to_device()
    @init.forward_sequential_method
    def __init__(self, input_shape, output_shape, outputs_are_between_0_and_1=False, hyperparams=None):
        super(GeneralApproximator, self).__init__()
        self.input_shape                 = input_shape
        self.output_shape                = output_shape
        self.outputs_are_between_0_and_1 = outputs_are_between_0_and_1
        
        input_size = product(self.input_shape)
        output_size = product(self.output_shape)
        
        big_size = max(input_size, output_size)
        number_of_middle_layers = (math.floor(math.log(math.sqrt(big_size))/math.log(4)) or 1) + 1 # log base 4 was hand picked with no data basis, 
            # big_size=10 => layers=0, 
            # big_size=1_000 => layers=2, 
            # big_size=50_000 => layers=3, 
            # big_size=500_000 => layers=4, 
            # big_size=1_500_000 => layers=5
        
        self.hyperparams = LazyDict(
            number_of_middle_layers=number_of_middle_layers,
            layer_shrink_amount=(input_size - output_size) / number_of_middle_layers,
            layer_sizes=None,
            loss_function=lambda current_outputs, correct_outputs: mse_loss(current_outputs, correct_outputs),
            optimizer_class=None,
            optimizer_kwargs=None,
        ).update(hyperparams or {})
        self.hyperparams.layer_sizes = self.hyperparams.layer_sizes or [input_size-(number*self.hyperparams.layer_shrink_amount) for number in range(0,self.hyperparams.number_of_middle_layers+1)]
        
        layers = Sequential(input_shape=self.input_shape)
        # ^ dynamically compute the output shape/size of layers (the nn.Linear below)
        layers.add_module('flatten' , nn.Flatten(start_dim=1, end_dim=-1))
        for each_index, each_size in enumerate(self.hyperparams.layer_sizes):
            layers.add_module(f'linear{each_index}' , nn.Linear(in_features=layers.output_size, out_features=math.ceil(each_size)))
            layers.add_module(f'relu{each_index}'   , nn.ReLU())
        layers.add_module('final_linear' , nn.Linear(in_features=layers.output_size, out_features=product(self.output_shape)))
        if self.outputs_are_between_0_and_1:
            layers.add_module('final_sigmoid', nn.Sigmoid())
        
        self.layers = layers
        self.optimizer = (self.hyperparams.optimizer_class or torch.optim.SGD)(
            self.parameters(),
            **LazyDict(
                lr=0.01,
            ).update(self.hyperparams.optimizer_kwargs or {}),
        )
    
    @convert_each_arg.to_tensor() # Use to_tensor(which_args=[0]) to only convert first arg
    @convert_each_arg.to_device() # Use to_device(which_args=[0]) to only convert first arg
    def fit(self, inputs, correct_outputs, epochs=1):
        for _ in range(epochs):
            self.optimizer.zero_grad()
            current_output = self.forward(inputs)
            loss = self.hyperparams.loss_function(current_output, correct_outputs)
            loss.backward()
            self.optimizer.step()
        return loss
    
    @convert_each_arg.to_tensor() # Use to_tensor(which_args=[0]) to only convert first arg
    @convert_each_arg.to_device() # Use to_device(which_args=[0]) to only convert first arg
    def predict(self, inputs):
        with torch.no_grad():
            return self.forward(inputs)

from sklearn.neighbors import NearestNeighbors
import numpy
class GeneralApproximator:
    def __init__(self, input_shape, output_shape, hyperparams=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.inputs = []
        self.outputs = []
        self.hyperparams = LazyDict(
            n_neighbors=3,
            algorithm='ball_tree',
        ).update(hyperparams or {})
        self.model = NearestNeighbors(**self.hyperparams)
    
    def preprocess(self, inputs):
        # flatten each input, 1 row = 1 input
        return torch.flatten(to_tensor(inputs), 1).numpy()
        
    def fit(self, inputs, correct_outputs):
        self.inputs += inputs
        self.outputs += correct_outputs
        self.model.fit(self.preprocess(inputs))
    
    def predict(self, inputs):
        # if no data has been gathered, just return random outputs
        # TODO: this should maybe show a warning
        if len(self.inputs) == 0:
            return torch.rand((len(inputs), self.output_shape))
        distances_for_inputs, indices_for_inputs = self.model.kneighbors( self.preprocess(inputs) )
        outputs = []
        for each_input, each_distances, each_indices in zip(inputs, distances_for_inputs, indices_for_inputs):
            distances = to_pure(each_distances)
            values    = tuple( to_pure(self.outputs[each_index]) for each_index in each_indices )
            values_as_array = to_tensor(values).numpy()
            average_value = numpy.mean(values_as_array, axis=0)
            outputs.append(average_value)
            # TODO: normalize and weight by distance
        return to_pure(outputs)


def test_the_approximator():
    input_shape = (2, 4)
    def generate_test_data_1(number_of_values=1000):
        inputs = []
        outputs = []
        import random
        for each in range(number_of_values):
            if each % 2:
                the_input = torch.ones(input_shape) + random.random()
                print(f'''the_input.shape = {the_input.shape}''')
                inputs.append(
                    the_input,
                )
                outputs.append(
                    to_tensor([ 1000 , -2]) * random.random(),
                )
            else:
                inputs.append(
                    torch.zeros(input_shape),
                )
                outputs.append(
                    to_tensor([ -1000, 2 ]) * random.random(),
                )
        return inputs, outputs

    inputs, correct_outputs = generate_test_data_1(100)
    # inputs, correct_outputs = generate_test_data_1(1000000)
    approximator = GeneralApproximator(input_shape=input_shape, output_shape=[1])
    approximator.fit(inputs=inputs, correct_outputs=correct_outputs)
    approximator.predict(inputs=inputs[0:4]) # approximator.predict(inputs=[[300,300,300,300]])
    correct_outputs[0:4]