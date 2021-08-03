import torch
import torch.nn as nn
from tools.basics import product, bundle

default_seed = 1
torch.manual_seed(default_seed)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# returns list of tensor sizes
def layer_output_shapes(input_shape, network):
    # convert lists to sequences
    if isinstance(network, list):
        network = nn.Sequential(*network)
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.network = network
        
        def forward(self, x):
            sizes = []
            for layer in self.network:
                x = layer(x)
                sizes.append(x.size())
            return sizes
    
    return Model().forward(torch.ones((1, *input_shape)))

def read_image(file_path):
    from PIL import Image
    import torchvision.transforms.functional as TF
    image = Image.open(file_path)
    return TF.to_tensor(image)

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
            
def onehot_argmax(tensor):
    tensor = to_tensor(tensor)
    the_max = max(each for each in tensor)
    onehot_tensor = torch.zeros_like(tensor)
    for each_index, each_value in enumerate(tensor):
        if each_value == the_max:
            onehot_tensor[each_index] = 1
    return onehot_tensor

def from_onehot_batch(tensor_batch):
    # make sure its a tensor
    tensor_batch = to_tensor(tensor_batch)
    return tensor_batch.max(1, keepdim=True).indices.squeeze()

def from_onehot(tensor):
    # make sure its a tensor
    tensor = to_tensor(tensor)
    return tensor.max(0, keepdim=True).indices.squeeze().item()

def batch_input_and_output(inputs, outputs, batch_size):
    from tools.basics import bundle
    batches = zip(bundle(inputs, batch_size), bundle(outputs, batch_size))
    for each_input_batch, each_output_batch in batches:
        yield to_tensor(each_input_batch), to_tensor(each_output_batch)

from simple_namespace import namespace

@namespace
def Network():
    
    def default_forward(self, input_data):
        """
        Uses:
            Self.layers
            Self.input_shape
            Self.output_shape
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
        input_data = to_tensor(input_data)
        
        # 
        # batch or not?
        # 
        if len(input_data.shape) == 3: 
            batch_size = None
            output_shape = self.output_shape
            # convert images into batches
            input_data = torch.reshape(input_data, (1, *input_data.shape))
        else:
            batch_size = tuple(input_data.shape)[0]
            output_shape = (batch_size, *self.output_shape)
        
        # TODO: consider the possibility of being givent a single 2D image
        
        # force into batches even if that means just adding a dimension
        from tools.basics import product
        batch_length = 1 if batch_size == None else batch_size
        input_data = torch.reshape(input_data, shape=(batch_length, *self.input_shape))
        input_data = input_data.type(torch.float)
        
        neuron_activations = input_data.to(self.device)
        for each_layer in self.layers:
            neuron_activations = each_layer(neuron_activations)
        
        # force the output to be the correct shape
        return torch.reshape(neuron_activations, output_shape)
    
    
    def default_setup(self, config):
        self.seed            = config.get("seed"           , default_seed)
        self.log_interval    = config.get("log_interval"   , 10)
        self.suppress_output = config.get("suppress_output", False)
        self.device          = config.get("device"         , torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.print = lambda *args, **kwargs: print(*args, **kwargs) if not self.suppress_output else None
        self.layers = nn.Sequential()
        
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
        number_correct = model_batch_output.eq(ideal_batch_output).sum().item()
        return number_correct
    
    def default_fit(self, *, input_output_pairs=None, dataset=None, loader=None, number_of_epochs=3, batch_size=64, shuffle=True):
        """
        Uses:
            self.update_weights(batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index)
            self.print(args)
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
        
        train_losses = []
        for epoch_index in range(number_of_epochs):
            self.train()
            for batch_index, (batch_of_inputs, batch_of_ideal_outputs) in enumerate(loader):
                loss = self.update_weights(batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index)
                from tools.basics import to_pure
                if batch_index % self.log_interval == 0:
                    self.print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}".format(
                            epoch_index+1,
                            batch_index * len(batch_of_inputs),
                            len(loader.dataset),
                            100.0 * batch_index / len(loader),
                            to_pure(loss),
                        )
                    )
                    train_losses.append(loss)
                    # TODO: add/allow checkpoints
                    # import os
                    # os.makedirs(f"{temp_folder_path}/results/", exist_ok=True)
                    # torch.save(self.state_dict(), f"{temp_folder_path}/results/model.pth")
                    # torch.save(self.optimizer.state_dict(), f"{temp_folder_path}/results/optimizer.pth")
        return train_losses
        
    def default_test(self, loader, correctness_function=None, loss_function=None):
        """
        Uses:
            self.forward(batch_of_inputs)
            self.print(args)
            self.eval() # provided by pytorch's `nn.Module`
            self.device # a pytorch device
        
        Optionally Uses:
            # returns the pytorch loss
            self.loss_function(batch_of_inputs, batch_of_ideal_outputs)
            # returns a number (number of correct guesses)
            self.correctness_function(batch_of_inputs, batch_of_ideal_outputs)
        """
        correctness_function = correctness_function or self.correctness_function
        loss_function = loss_function or self.loss_function
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_of_inputs, batch_of_ideal_outputs in loader:
                actual_output = self.forward(batch_of_inputs)
                batch_of_inputs = actual_output.type(torch.float).to(self.device)
                batch_of_ideal_outputs = batch_of_ideal_outputs.type(torch.float).to(self.device)
                test_loss += loss_function(batch_of_inputs, batch_of_ideal_outputs)
                correct += correctness_function(batch_of_inputs, batch_of_ideal_outputs)
        
        # convert to regular non-tensor data
        test_loss = test_loss.item()
        test_loss /= len(loader.dataset)
        self.print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(loader.dataset),
                100.0 * correct / len(loader.dataset),
            )
        )
        return correct
    
    return locals()

_image_log_count = 0
def log_image(image_tensor):
    global _image_log_count
    import torchvision.transforms.functional as F
    import os
    
    _image_log_count += 1
    os.makedirs("./logs.dont-sync", exist_ok=True)
    image_path = f"./logs.dont-sync/display_{_image_log_count}.png"
    F.to_pil_image(image_tensor).save(image_path)
    print("image logged: "+image_path)
