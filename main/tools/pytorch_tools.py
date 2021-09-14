#%% pytorch_tools
import torch
import torch.nn as nn
from tools.basics import product, bundle
from tools.record_keeper import RecordKeeper
#%% pytorch_tools

default_seed = 1
torch.manual_seed(default_seed)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# returns list of tensor sizes
def layer_output_shapes(network, input_shape=None):
    # convert OrderedDict's to just lists
    from collections import OrderedDict
    if isinstance(network, OrderedDict):
        network = list(network.values())
    # convert lists to sequences
    if isinstance(network, list):
        network = nn.Sequential(*network)
    
    # run a forward pass to figure it out
    neuron_activations = torch.ones((1, *input_shape))
    sizes = []
    for layer in network:
        # if its not a loss function
        if not isinstance(layer, torch.nn.modules.loss._Loss):
            neuron_activations = layer(neuron_activations)
            sizes.append(neuron_activations.size())
    
    return sizes

def read_image(file_path):
    from PIL import Image
    import torchvision.transforms.functional as TF
    image = Image.open(file_path)
    return TF.to_tensor(image)

def tensor_to_image(tensor):
    import torchvision.transforms.functional as TF
    return TF.to_pil_image(tensor)

def to_tensor(an_object):
    from tools.basics import is_iterable
    
    # if already a tensor, just return
    if isinstance(an_object, torch.Tensor):
        return an_object
        
    # if scalar, wrap it with a tensor
    if not is_iterable(an_object):
        return torch.tensor(an_object)
    else:
        as_list = tuple([ each for each in an_object ])
        
        # # check for all-scalar container
        # is_all_scalar = True
        # for each in as_list:
        #     if is_iterable(each):
        #         is_all_scalar = False
        #         break
        # if is_all_scalar:
        #     return torch.tensor(as_list)
        
        size_mismatch = False
        biggest_number_of_dimensions = 0
        non_one_dimensions = None
        converted_data = []
        # check the shapes of everything
        for each in as_list:
            tensor = to_tensor(each)
            converted_data.append(tensor)
            skipping = True
            each_non_one_dimensions = []
            for index, each_dimension in enumerate(tensor.shape):
                # keep track of number of dimensions
                if index+1 > biggest_number_of_dimensions:
                    biggest_number_of_dimensions += 1
                    
                if each_dimension != 1:
                    skipping = False
                if skipping and each_dimension == 1:
                    continue
                else:
                    each_non_one_dimensions.append(each_dimension)
            
            # if uninitilized
            if non_one_dimensions is None:
                non_one_dimensions = list(each_non_one_dimensions)
            # if dimension already exists
            else:
                # make sure its the correct shape
                if non_one_dimensions != each_non_one_dimensions:
                    size_mismatch = True
                    break
        
        if size_mismatch:
            sizes = "\n".join([ f"    {tuple(to_tensor(each).shape)}" for each in as_list])
            raise Exception(f'When converting an object to a torch tensor, there was an issue with the shapes not being uniform. All shapes need to be the same, but instead the shapes were:\n {sizes}')
        
        # make all the sizes the same by filling in the dimensions with a size of one
        reshaped_list = []
        for each in converted_data:
            shape = tuple(each.shape)
            number_of_dimensions = len(shape)
            number_of_missing_dimensions = biggest_number_of_dimensions - number_of_dimensions 
            missing_dimensions_tuple = (1,)*number_of_missing_dimensions
            reshaped_list.append(torch.reshape(each, (*missing_dimensions_tuple, *shape)))
        
        return torch.stack(reshaped_list).type(torch.float)
            
class OneHotifier():
    def __init__(self, possible_values):
        # convert to tuple if needed
        if not hasattr(possible_values, "__len__"):
            possible_values = tuple(possible_values)
        self.possible_values = possible_values
    
    def to_onehot(self, value):
        index = self.possible_values.index(value)
        return torch.nn.functional.one_hot(
            torch.tensor(index),
            len(self.possible_values)
        )
    
    def from_onehot(self, vector):
        vector = to_tensor(vector)
        index_value = vector.max(0).indices
        return self.possible_values[index_value]
        
def onehot_argmax(tensor):
    tensor = to_tensor(tensor)
    the_max = max(each for each in tensor)
    onehot_tensor = torch.zeros_like(tensor)
    for each_index, each_value in enumerate(tensor):
        if each_value == the_max:
            onehot_tensor[each_index] = 1
    return onehot_tensor

def from_onehot_batch(tensor_batch):
    device = None
    if isinstance(tensor_batch, torch.Tensor):
        device = tensor_batch.device
    # make sure its a tensor
    tensor_batch = to_tensor(tensor_batch)
    output = tensor_batch.max(1, keepdim=True).indices.squeeze()
    # send to same device
    return output.to(device) if device else output

def from_onehot(tensor):
    # make sure its a tensor
    tensor = to_tensor(tensor)
    return tensor.max(0, keepdim=True).indices.squeeze().item()

def batch_input_and_output(inputs, outputs, batch_size):
    from tools.basics import bundle
    batches = zip(bundle(inputs, batch_size), bundle(outputs, batch_size))
    for each_input_batch, each_output_batch in batches:
        yield to_tensor(each_input_batch), to_tensor(each_output_batch)

def unnormalize(mean, std, image):
    import torchvision.transforms as transforms
    normalizer = transforms.Normalize((-mean / std), (1.0 / std))
    return normalizer(image)


from simple_namespace import namespace

@namespace
def Network():
    
    def default_forward(self, input_data):
        """
        Uses:
            self.device
            self.input_shape
            self.output_shape
        Arguments:
            input_data:
                either an input image or batch of images
                should be a torch tensor with a shape of (batch_size, channels, height, width)
        Ouptut:
            a torch tensor the shape of the latent space
        Examples:
            obj.forward(torch.tensor([
                # first image in batch
                [
                    # red layer
                    [
                        [ 1, 2, 3 ],
                        [ 4, 5, 6] 
                    ], 
                    # blue layer
                    [
                        [ 1, 2, 3 ],
                        [ 4, 5, 6] 
                    ], 
                    # green layer
                    [
                        [ 1, 2, 3 ],
                        [ 4, 5, 6] 
                    ],
                ] 
            ]))
        
        """
        # converts to torch if needed
        input_data = to_tensor(input_data).type(torch.float).to(self.device)
        
        # 
        # batch or not?
        # 
        is_a_batch = len(input_data.shape) > len(self.input_shape)
        if not is_a_batch: 
            batch_size = 1
            # convert images into batches
            input_data = torch.reshape(input_data, (1, *input_data.shape))
            output_shape = self.output_shape
        else:
            batch_size = tuple(input_data.shape)[0]
            output_shape = (batch_size, *self.output_shape)
        
        # 
        # forward pass
        # 
        neuron_activations = input_data
        for each_layer in self.children():
            # if its not a loss function
            if not isinstance(each_layer, torch.nn.modules.loss._Loss):
                neuron_activations = each_layer(neuron_activations)
        
        # force the output to be the correct shape
        return torch.reshape(neuron_activations, output_shape)
    
    def default_setup(self, config):
        # check for pytorch lightning
        try:
            import pytorch_lightning as pl
            LightningModule = pl.LightningModule
            Trainer = pl.Trainer
        except Exception as error:
            LightningModule = None
            Trainer = None
        
        self.setup_config    = config
        self.seed            = config.get("seed"           , default_seed)
        self.suppress_output = config.get("suppress_output", False)
        self.log_interval    = config.get("log_interval"   , 10)
        self.hardware        = config.get("device"         , torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.record_keeper   = config.get("record_keeper"  , RecordKeeper()).sub_record_keeper(model=self.__class__.__name__)
        self.show = lambda *args, **kwargs: print(*args, **kwargs) if not self.suppress_output else None
        self.to(self.hardware)
        if not isinstance(self, LightningModule):
            self.device = self.hardware
        else:
            self._is_lightning_module = True
            self.new_trainer = lambda *args, **kwargs: Trainer(*args, **{
                # default values
                **({
                    "gpus": torch.cuda.device_count(),
                    "auto_select_gpus": True,
                } if torch.cuda.device_count() > 0 else {}),
                **kwargs,
            })
    
    def default_update_record_keepers(self):
        self.setup_config
    
    def default_update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        """
        Uses:
            self.optimizer # pytorch optimizer class
            self.forward(batch_of_inputs)
            self.loss_function(batch_of_actual_outputs, batch_of_ideal_outputs)
        """
        self.optimizer.zero_grad()
        batch_of_actual_outputs = self.forward(batch_of_inputs)
        loss = self.loss_function(batch_of_actual_outputs, batch_of_ideal_outputs)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def onehot_correctness_function(self, model_batch_output, ideal_batch_output):
        """
        Summary:
            This assumes both the output of the network and the output of the dataset
            are one-hot encoded.
        """
        # convert to a batch of real-numbered outputs
        model_batch_output = from_onehot_batch(model_batch_output)
        ideal_batch_output = from_onehot_batch(ideal_batch_output)
        # element-wise compare how many are equal, then sum them up into a scalar
        model_batch_output = model_batch_output.to(self.hardware)
        ideal_batch_output = ideal_batch_output.to(self.hardware)
        number_correct = model_batch_output.eq(ideal_batch_output).sum().item()
        return number_correct
    
    def default_fit(self, *, input_output_pairs=None, dataset=None, loader=None, batch_size=64, shuffle=True, **kwargs):
        """
        Uses:
            self.update_weights(batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index)
            self.show(args)
            self.train() # provided by pytorch's `nn.Module`
        
        Examples:
            model.fit(
                dataset=torchvision.datasets.MNIST(<mnist args>),
                epochs=4,
                batch_size=64,
            )
            
            model.fit(
                loader=torch.utils.data.DataLoader(<dataloader args>),
                epochs=4,
            )
        """
        # TODO: test input_output_pairs
        if input_output_pairs is not None:
            # creates batches
            def bundle(iterable, bundle_size):
                next_bundle = []
                for each in iterable:
                    next_bundle.append(each)
                    if len(next_bundle) == bundle_size:
                        yield tuple(next_bundle)
                        next_bundle = []
                # return any half-made bundles
                if len(next_bundle) > 0:
                    yield tuple(next_bundle)
            # unpair, batch, then re-pair the inputs and outputs
            input_generator        = (each for each, _ in input_output_pairs)
            ideal_output_generator = (each for _   , each in input_output_pairs)
            seperated_batches = zip(bundle(input_generator, batch_size), bundle(ideal_output_generator, batch_size))
            loader = ((to_tensor(each_input_batch), to_tensor(each_output_batch)) for each_input_batch, each_output_batch in seperated_batches)
            # NOTE: shuffling isn't possible when there is no length (and generators don't have lengths). So maybe think of an alternative
        else:
            # convert the dataset into a loader (assumming loader was not given)
            if isinstance(dataset, torch.utils.data.Dataset):
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                )
        
        if hasattr(self, "_is_lightning_module"):
            self.prev_trainer = self.new_trainer(**kwargs)
            output = self.prev_trainer.fit(self, loader)
            # go back to the hardware to unto the changes made by pytorch lightning
            self.to(self.hardware)
            return output
        else:
            train_losses = []
            for epoch_index in range(kwargs.get("max_epochs", 1)):
                self.train()
                for batch_index, (batch_of_inputs, batch_of_ideal_outputs) in enumerate(loader):
                    loss = self.update_weights(batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index)
                    from tools.basics import to_pure
                    if batch_index % self.log_interval == 0:
                        count = batch_index * len(batch_of_inputs)
                        total = len(loader.dataset)
                        pure_loss = to_pure(loss)
                        self.show(f"\r[Train]: epoch: {epoch_index:>4}, batch: {count:>10}/{total}", sep='', end='', flush=True)
                        train_losses.append(loss)
                        # TODO: add/allow checkpoints
            self.show()
            return train_losses
        
    def default_test(self, loader, correctness_function=None, loss_function=None):
        """
        Uses:
            self.forward(batch_of_inputs)
            self.show(args)
            self.eval() # provided by pytorch's `nn.Module`
            self.hardware # a pytorch device
        
        Optionally Uses:
            # returns the pytorch loss
            self.loss_function(batch_of_inputs, batch_of_ideal_outputs)
            # returns a number (number of correct guesses)
            self.correctness_function(batch_of_inputs, batch_of_ideal_outputs)
        """
        correctness_function = correctness_function or self.correctness_function
        loss_function = loss_function or self.loss_function
        self.eval()
        test_loss_accumulator = 0
        correct_count = 0
        with torch.no_grad():
            for batch_of_inputs, batch_of_ideal_outputs in loader:
                actual_output = self.forward(batch_of_inputs)
                actual_output = actual_output.type(torch.float).to(self.device)
                batch_of_ideal_outputs = batch_of_ideal_outputs.type(torch.float).to(self.device)
                test_loss_accumulator += loss_function(actual_output, batch_of_ideal_outputs)
                correct_count += correctness_function(actual_output, batch_of_ideal_outputs)
        
        # convert to regular non-tensor data
        from tools.basics import to_pure
        sample_count = len(loader.dataset)
        accuracy     = correct_count / len(loader.dataset)
        average_loss = to_pure(test_loss_accumulator) / sample_count
        if hasattr(self, "record_keeper"):
            self.record_keeper.commit_record(additional_info=dict(
                testing=True,
                average_loss=average_loss,
                accuracy=correct_count / sample_count,
                correct=correct_count,
            ))
        
        self.show(f"[Test]: average_loss: {average_loss:>9.4f}, accuracy: {accuracy:>4.2f}, {correct_count}/{sample_count:.0f}")
        return correct_count
    
    return locals()

_image_log_count = 0
def log_image(image_tensor):
    global _image_log_count
    import torchvision.transforms.functional as F
    import os
    
    _image_log_count += 1
    os.makedirs("./logs.do_not_sync", exist_ok=True)
    image_path = f"./logs.do_not_sync/display_{_image_log_count}.png"
    F.to_pil_image(image_tensor).save(image_path)
    print("image logged: "+image_path)

#%% 