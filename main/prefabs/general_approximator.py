import math
import json

import torch
import torch.nn as nn
from blissful_basics import print
from trivial_torch_tools import Sequential, init, convert_each_arg, to_tensor
from trivial_torch_tools.generics import product, to_pure, flatten
from super_map import LazyDict
from super_hash import super_hash

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

import numpy
numpy.float = float # workaround for DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
from sklearn.neighbors import NearestNeighbors
class GeneralApproximator:
    def __init__(self, input_shape, output_shape, max_number_of_points=None, enable_exact_match_cache=True, hyperparams=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.max_number_of_points = max_number_of_points
        self.enable_exact_match_cache = enable_exact_match_cache
        self.inputs = None
        self.outputs = None
        self.hyperparams = LazyDict(
            n_neighbors=3,
            algorithm='ball_tree',
        ).update(hyperparams or {})
        self._recently_used_indicies = []
        if self.enable_exact_match_cache:
            self._cache = {}
    
    def preprocess(self, inputs):
        return to_tensor([flatten(each) for each in inputs]).numpy()
        
    def fit(self, inputs, correct_outputs):
        preprocessed_inputs  = self.preprocess(inputs)
        preprocessed_outputs = self.preprocess(correct_outputs)
        self._update_exact_match_cache(preprocessed_inputs, preprocessed_outputs) # always have the cache use the latest values
        self.inputs  = preprocessed_inputs  if type(self.inputs ) == type(None) else numpy.concatenate((self.inputs , preprocessed_inputs ), axis=0)
        self.outputs = preprocessed_outputs if type(self.outputs) == type(None) else numpy.concatenate((self.outputs, preprocessed_outputs), axis=0)
        if self.max_number_of_points:
            half_max = int(self.max_number_of_points/2)
            
            # reuse useful ones
            self._recently_used_indicies = self._recently_used_indicies[-half_max:]
            if self._recently_used_indicies:
                values = tuple(self.inputs[ each_index] for each_index in self._recently_used_indicies)
                recently_used_inputs  = numpy.stack(tuple(self.inputs[ each_index] for each_index in self._recently_used_indicies), axis=0)
                recently_used_outputs = numpy.stack(tuple(self.outputs[each_index] for each_index in self._recently_used_indicies), axis=0)
                
                # drop old points that have not been used recently for prediction
                self.inputs  = numpy.concatenate((self.inputs,  recently_used_inputs ), axis=0)[-self.max_number_of_points:]
                self.outputs = numpy.concatenate((self.outputs, recently_used_outputs), axis=0)[-self.max_number_of_points:]
            else:
                self.inputs  = self.inputs[-self.max_number_of_points:]
                self.outputs = self.outputs[-self.max_number_of_points:]
        self.model = NearestNeighbors(**self.hyperparams).fit(self.inputs)
    
    def predict(self, inputs):
        # if no data has been gathered, just return zeros
        # TODO: this should maybe show a warning
        if type(self.inputs) == type(None) or len(self.inputs) <= self.hyperparams.n_neighbors:
            return torch.zeros((len(inputs), *self.output_shape))
        
        # 
        # cache check
        # 
        preprocessed_inputs = self.preprocess(inputs)
        if self.enable_exact_match_cache:
            outputs = []
            for each in preprocessed_inputs:
                each_hash = hash(json.dumps(to_pure(each)))
                if each_hash not in self._cache:
                    break
                outputs.append(self._cache[each_hash])
            if len(outputs) == len(inputs):
                return outputs
        
        distances_for_inputs, indices_for_inputs = self.model.kneighbors(preprocessed_inputs)
                
        outputs = []
        for each_input, each_distances, neighbor_indices in zip(inputs, distances_for_inputs, indices_for_inputs):
            self._update_used_indicies(neighbor_indices)
            distances = to_pure(each_distances)
            with print.indent.block("kneighbors", disable=True):
                print(LazyDict({
                    loop_index: LazyDict(
                        each_distance=each_distance,
                        neighbor_index=each_neighbor_index,
                        input=to_pure(self.inputs[each_neighbor_index]),
                        output=self.outputs[each_neighbor_index]
                    )
                        for loop_index, (each_neighbor_index, each_distance) in enumerate(zip(neighbor_indices, distances))
                }))
            
            # if there is an exact overlap, dont average the neighbors
            exact_match = False
            for neighbor_index, each_distance in zip(neighbor_indices, distances):
                if each_distance == 0:
                    exact_match = True
                    break
            if exact_match:
                output = to_pure(self.outputs[neighbor_index])
                outputs.append(output)
                continue
            else:
                min_contribution = 0.1 # 0.1=>10%
                # equivlent to:
                    # distance_weights = [ (1/(distance**2)) for distance in distances ]
                    # normalized_weights = [ each+min_contribution for each in self._normalize(distance_weights)]
                    # total = sum(normalized_weights)
                    # sum_to_1_weights = [ each/total for each in normalized_weights ]
                    
                distance_weights   = 1/(each_distances ** 2)
                normalized_weights = self.normalize_numpy_array(distance_weights) + min_contribution
                values = tuple(self.outputs[each_index] for each_index in neighbor_indices )
                values_as_array = numpy.concatenate(values, axis=0)
                # average value often isnt a scalar because of self.output_shape
                average_value = numpy.average(values_as_array, axis=0, weights=normalized_weights)
                outputs.append(average_value)
                
        return to_pure(outputs)
    
    def _update_exact_match_cache(self, inputs, outputs):
        if self.enable_exact_match_cache:
            for each_input, each_output in zip(inputs, outputs):
                hash_value = hash(json.dumps(to_pure(each_input)))
                self._cache[hash_value] = each_output
    
    def _update_used_indicies(self, indicies):
        # only keep track if trucation is going to happen
        if self.max_number_of_points:
            for each in indicies:
                try:
                    self._recently_used_indicies.remove(each)
                except ValueError:
                    self._recently_used_indicies.append(each)
    
    @staticmethod
    def normalize_numpy_array(array):
        import numpy
        the_max = array.max()
        the_min = array.min()
        if the_max == the_min:
            return numpy.zeros_like(array)
        else:
            the_range = the_max - the_min
            return (array - the_min)/the_range


def test_the_approximator():
    input_shape = (2, 4)
    def generate_test_data_1(number_of_values=1000):
        inputs = []
        outputs = []
        import random
        for each in range(number_of_values):
            if each % 2:
                the_input = torch.ones(input_shape) + random.random()
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