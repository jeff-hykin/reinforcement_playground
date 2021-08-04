#%% 
# basics
# 
if True:
    from tools.basics import *
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    from tools.file_system_tools import FS
    from torchvision import transforms
    import torch.nn.functional as F
    from torch.optim.optimizer import Optimizer, required
    # randomize the torch seed
    from time import time as now
    right_now = now()
    torch.manual_seed(right_now)
    print('manual_seed: ', right_now)


#%% 
# pytorch_tools
# 
if True:
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
    
    class ImageModelSequential(nn.Module):
        @property
        def setup(self):
            """
            Example:
                with self.setup(input_shape=None, output_shape=None, loss_function=None, layers=None, **config):
                    # normal setup stuff
                    self.layers.add_module("layer1", nn.Linear(self.size_of_last_layer, int(self.input_feature_count/2)))
                    self.layers.add_module("layer1_activation", nn.ReLU())
                    self.layers.add_module("layer2", nn.Linear(self.size_of_last_layer, self.output_feature_count))
                    self.layers.add_module("layer2_activation", nn.Sigmoid())
                    
                    # default to squared error loss_function
                    self.loss_function = loss_function or (lambda input_batch, ideal_output_batch: torch.mean((self.forward(input_batch) - ideal_output_batch)**2))
                    
            Arguments:
                input_shape:
                    a tuple that expected to be (image_channels, image_height, image_width)
                    where image_channels, image_height, and image_width are all integers
                    
                output_shape:
                    a tuple, probably with only one large number, e.g. (32, 1) or (32, 1, 1)
                    which lets you pick the shape of your output
                    more dynamic output shapes are allowed too, e.g (32, 32)
                
                loss_function:
                    Arguments:
                        input_batch:
                            a torch tensor of images with shape (batch_size, channels, height, width)
                        ideal_output_batch:
                            a vector of latent spaces with shape (batch_size, *output_shape) 
                            for example if the output_shape was (32, 16) then this would be (batch_size, 32, 16)
                    Note:
                        must perform only pytorch tensor operations on the input_batch
                        (see here for vaild pytorch tensor operations: https://towardsdatascience.com/how-to-train-your-neural-net-tensors-and-autograd-941f2c4cc77c)
                        otherwise pytorch won't be able to perform backpropogation
                    Ouput:
                        should return a torch tensor that is the result of an operation with the input_batch
                        
            """
            # the reason this is kind of janky is so that we can perform checks/code at the end of the init 
            # like sending the model to the device, and making sure the input/output shape is right
            real_super = super
            class Setup(object):
                def __init__(_, input_shape=None, output_shape=None, loss=None, optimizer=None, layers=None, **config):
                    # 
                    # basic setup
                    # 
                    real_super(ImageModelSequential, self).__init__()
                    self.suppress_output = config.get("suppress_output", False)
                    self.print = lambda *args, **kwargs: print(*args, **kwargs) if not self.suppress_output else None
                    # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.device = torch.device('cpu') # FIXME: I had to do this to get the loss function working
                    
                    self.layers = layers or nn.Sequential()
                    self.loss = loss
                    self.optimizer = optimizer
                    # 
                    # upgrade image input to 3D if 2D
                    # 
                    if len(input_shape) == 2: input_shape = (1, *input_shape)
                    # channels, height, width  = input_shape
                    self.input_shape = input_shape
                    self.output_shape = output_shape
                    
                    self.input_feature_count = product(self.input_shape)
                    self.output_feature_count = product(self.output_shape)
                
                def __enter__(_, *args, **kwargs):
                    pass
            
                def __exit__(_, *args, **kwargs):
                    # TODO: check that there is at least one layer
                    # TODO: check that the input/output layer have the right input/output sizes/shapes
                    self.to(self.device)
            
            return Setup
            
    
        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, input_data):
            """
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
        
        @property
        def size_of_last_layer(self):
            # if no layers, then use the input size
            if len(self.layers) == 0:
                return self.input_feature_count
            else:
                return product(layer_output_shapes(self.input_shape, self.layers)[-1])
        
        def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
            self.optimizer.zero_grad()
            batch_of_actual_outputs = self.forward(batch_of_inputs)
            loss = self.loss_function(batch_of_actual_outputs, batch_of_ideal_outputs)
            loss.backward()
            self.optimizer.step()
            return loss
        
        def fit(self, *, input_output_pairs=None, dataset=None, loader=None, number_of_epochs=3, batch_size=64, shuffle=True):
            """
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
        
        def test(self, test_loader):
            from tools.basics import max_index
            from tools.pytorch_tools import onehot_argmax
            # TODO: change this to use the full loss instead of exact equivlence
            test_losses = []
            self.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for batch_of_inputs, batch_of_ideal_outputs in test_loader:
                    actual_output = self.forward(batch_of_inputs)
                    test_loss += self.loss_function(actual_output.type(torch.float), batch_of_ideal_outputs.type(torch.float)).item()
                    correct += sum(
                        1 if torch.equal(onehot_argmax(each_output).float(), onehot_argmax(each_ideal_output).float()) else 0
                            for each_output, each_ideal_output in zip(actual_output.float(), batch_of_ideal_outputs.float())
                    )
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print(
                "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    100.0 * correct / len(test_loader.dataset),
                )
            )
            self.test_losses = test_losses
            return correct
        
        def compute_gradients_for(self, input_batch, ideal_outputs_batch, loss_function=None, retain_graph=False):
            """
            Examples:
                (weight_gradient_layer1, bias_gradient_layer1), *other_layer_gradients = compute_gradients_for(
                    input_batch=input_batch,
                    ideal_outputs_batch=ideal_outputs_batch,
                )
            Output:
                a list, that contains tuples
                each tuple contains the weight gradient (number), followed by the bias gradient
                if a layer is missing one of those, then it will be None
                if a layer is missing both of those, it is skipped
            """
            # FIXME: implment
            pass

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

#%% 
# Dataset tools
# 
if True:
    import gdown
    import os
    import torch
    temp_folder_path = f"{os.environ.get('PROJECTR_FOLDER')}/settings/.cache/common_format_datasets"

    class SimpleDataset(torch.utils.data.Dataset):
        """
        When inheriting from this, make sure to define the following
            __init__(self, **kwargs):
                super(self.__class__, self).__init__(**kwargs)
            
            __len__(self):
            
            get_input(self, index):
            
            get_output(self, index):
        """
        @property
        def input_shape(self):
            input_1, _ = self[0]
            return tuple(input_1.shape)

        @property
        def output_shape(self):
            _, output_1 = self[0]
            return tuple(output_1.shape)
        
        def __init__(self, transform_input=None, transform_output=None):
            # save these for later
            self.transform_input  = transform_input
            self.transform_output = transform_output
            
        def __getitem__(self, index):
            # 
            # input
            # 
            an_input = self.get_input(index)
            if self.transform_input:
                an_input = self.transform_input(an_input)
            # 
            # output
            # 
            corrisponding_output = self.get_output(index)
            if self.transform_output:
                corrisponding_output = self.transform_output(corrisponding_output)
            
            # return
            return an_input, corrisponding_output

    # FIXME: the images are corrupt for some reason
    class Mnist(SimpleDataset):
        """
        file structure:
            $DATASET_FOLDER/
                mnist/
                    img_0/
                        data.jpg
                        data.integer
                    img_1/
                        data.jpg
                        data.integer
                    ...
        
        """
        number_of_classes = 10
        _path = None
        @classmethod
        def get_path(cls, quiet=True):
            """
            returns the mnist folder, downloading the mnist data if needed
            """
            if cls._path is None:
                url = 'https://drive.google.com/uc?id=1im9_agM-hHl8dKQU5uWh054rScjPINtz' # jeff hykin's google drive copy of mnist
                output = f'{temp_folder_path}/mnist.zip'
                gdown.cached_download(url, output, postprocess=gdown.extractall, quiet=quiet)
                
                cls._path = temp_folder_path+"/mnist"
            return cls._path
            
        def __init__(self, **kwargs):
            super(Mnist, self).__init__(**kwargs)
            self.folder_path = self.get_path()
            
            from tools.file_system_tools import FS
            # ignore hidden files/folders (start with a .)
            self.ids = [ each for each in FS.list_folders(self.folder_path) if each[0] != '.' ]

        def __len__(self):
            return len(self.ids)
            
        def get_input(self, index):
            from torchvision.io import read_image
            item_folder =  self.folder_path +"/"+ self.ids[index]
            return read_image(item_folder+"/data.jpg")

        def get_output(self, index):
            from tools.file_system_tools import FS
            from tools.pytorch_tools import to_tensor
            import torch.nn.functional as F
            
            item_folder =  self.folder_path +"/"+ self.ids[index]
            return int(FS.read(item_folder+"/data.integer"))

    mnist_dataset = Mnist()


    def binary_mnist(numbers):
        import torchvision
        class Dataset(torchvision.datasets.MNIST):
            number_of_classes = 10
            def __init__(self, *args, **kwargs):
                super(Dataset, self).__init__(*args, **kwargs)
            def __getitem__(self, index):
                an_input, corrisponding_output = super(Dataset, self).__getitem__(index)
                if corrisponding_output in numbers:
                    return an_input, torch.tensor([1,0])
                else:
                    return an_input, torch.tensor([0,1])

        from tools.basics import temp_folder
        options = dict(
            root=f"{temp_folder}/files/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        from torchsampler import ImbalancedDatasetSampler

        # 1/6th of the data is for testing
        dataset = Dataset(**options)
        number_of_splits = 6
        test_sections = 1
        number_of_test_elements = int(test_sections * (len(dataset) / 6))
        number_of_train_elements = len(dataset) - number_of_test_elements
        train_dataset, test_dataset = torch.utils.data.random_split(Dataset(**options), [number_of_train_elements, number_of_test_elements])


        from collections import Counter
        total_number_of_samples = len(train_dataset)
        class_counts = dict(Counter(tuple(each_output.tolist()) for each_input, each_output in train_dataset))
        class_weights = { each_class_key: total_number_of_samples/each_value for each_class_key, each_value in class_counts.items() }
        weights = [ class_weights[tuple(each_output.tolist())] for each_input, each_output in train_dataset ]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), int(total_number_of_samples))

        # test_dataset = Dataset(**{**options, "train":False})
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            # ImbalancedDatasetSampler(train_dataset, callback_get_label=lambda *args:range(len(train_dataset))),
            batch_size=64,
            # shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            # sampler=ImbalancedDatasetSampler(test_dataset, callback_get_label=lambda *args:range(len(test_dataset))),
            # sampler=ImbalancedDatasetSampler(test_dataset),
            batch_size=1000,
            # shuffle=True,
        )
        return train_dataset, test_dataset, train_loader, test_loader

#%% 
if True:
    __file__ = '/home/jeffhykin/repos/reinforcement_playground/main/agents/informed_vae/mnist_test.py'

    class SimpleClassifier(nn.Module):
        def __init__(self, **config):
            super(SimpleClassifier, self).__init__()
            # 
            # options
            # 
            Network.default_setup(self, config)
            self.input_shape     = config.get("input_shape"    , (1, 28, 28))
            self.output_shape    = config.get("output_shape"   , (2,))
            self.batch_size      = config.get("batch_size"     , 64  )
            self.test_batch_size = config.get("test_batch_size", 1000)
            self.epochs          = config.get("epochs"         , 3   )
            self.lr              = config.get("lr"             , 0.01)
            self.momentum        = config.get("momentum"       , 0.5 )
            
            # 
            # layers
            # 
            # 1 input image, 10 output channels, 5x5 square convolution kernel
            self.layers.add_module("conv1", nn.Conv2d(1, 10, kernel_size=5))
            self.layers.add_module("conv1_pool", nn.MaxPool2d(2))
            self.layers.add_module("conv1_activation", nn.ReLU())
            self.layers.add_module("conv2", nn.Conv2d(10, 10, kernel_size=5))
            self.layers.add_module("conv2_drop", nn.Dropout2d())
            self.layers.add_module("conv2_pool", nn.MaxPool2d(2))
            self.layers.add_module("conv2_activation", nn.ReLU())
            self.layers.add_module("flatten", nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
            self.layers.add_module("fc1", nn.Linear(self.size_of_last_layer, product(self.output_shape)))
            self.layers.add_module("fc1_activation", nn.LogSoftmax(dim=1))
            
            # 
            # support (optimizer, loss)
            # 
            self.to(self.device)
            # create an optimizer
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        
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




#%% 
# test_split_network
# 
if True:

    # from agents.informed_vae.simple_classifier import SimpleClassifier
    # 
    # train and test
    # 
    split = SplitAutoEncoder()
    classifier = SimpleClassifier()
    results = []
    for each in [9]:
        result = {}
        train_dataset, test_dataset, train_loader, test_loader = quick_loader(binary_mnist([each]), [5, 1])
        
        # 
        # split
        # 
        # reset the task network part (last few layers)
        split.task_network = nn.Sequential(
            nn.Linear(product(split.latent_shape), 2), # binary classification
            nn.Sigmoid(),
        )
        result["split_train"] = split.fit(loader=train_loader, number_of_epochs=3)
        result["split_test"] = split.test(test_loader)
        
        # 
        # classifier
        # 
        train_dataset, test_dataset, train_loader, test_loader = binary_mnist([each])
        # reset the task network part (last few layers)
        classifier.task_network = nn.Sequential(
            nn.Linear(product(classifier.latent_shape), 2), # binary classification
            nn.Sigmoid(),
        )
        result["classifier_train"] = classifier.fit(loader=train_loader, number_of_epochs=3)
        result["classifier_test"] = classifier.test(test_loader)
        
        # 
        # fresh_classifier
        # 
        fresh_classifier = simple_classifier()
        result["fresh_classifier_train"] = fresh_classifier.fit(loader=train_loader, number_of_epochs=3)
        result["fresh_classifier_test"] = fresh_classifier.test(test_loader)
        
        # save
        results.append(result)

    for each_iter in results:
        result_string = "results: "
        for each_key, each_network_result in each_iter.items():
            if "test" in each_key:
                result_string += f'({each_key.replace("test", "")}: {each_network_result})  '
        print(result_string)
#%% 
# Working Network
# 
if True:
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from tools.basics import product
    
    class SamNet(nn.Module):
        def __init__(self, **config):
            super(SamNet, self).__init__()
            # 
            # options
            # 
            Network.default_setup(self, config)
            self.input_shape     = config.get("input_shape"    , (1, 28, 28))
            self.output_shape    = config.get("output_shape"   , (2,))
            self.batch_size      = config.get("batch_size"     , 64  )
            self.test_batch_size = config.get("test_batch_size", 1000)
            self.epochs          = config.get("epochs"         , 3   )
            self.lr              = config.get("lr"             , 0.01)
            self.momentum        = config.get("momentum"       , 0.5 )
            
            # 
            # layers
            # 
            # 1 input image, 10 output channels, 5x5 square convolution kernel
            self.layers.add_module("conv1", nn.Conv2d(1, 10, kernel_size=5))
            self.layers.add_module("conv1_pool", nn.MaxPool2d(2))
            self.layers.add_module("conv1_activation", nn.ReLU())
            self.layers.add_module("conv2", nn.Conv2d(10, 10, kernel_size=5))
            self.layers.add_module("conv2_drop", nn.Dropout2d())
            self.layers.add_module("conv2_pool", nn.MaxPool2d(2))
            self.layers.add_module("conv2_activation", nn.ReLU())
            self.layers.add_module("flatten", nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
            self.layers.add_module("fc1", nn.Linear(self.size_of_last_layer, product(self.output_shape)))
            self.layers.add_module("fc1_activation", nn.LogSoftmax(dim=1))
            
            # 
            # support (optimizer, loss)
            # 
            self.to(self.device)
            # create an optimizer
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        
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
        
        #     """
        #     You just have to define the forward function, and the backward function
        #     (where gradients are computed) is automatically defined for you using
        #     autograd. You can use any of the Tensor operations in the
        #     forward function.
        #     """
        #     x = x.to(self.device)
        #     # Max pooling over a 2x2 window
        #     x = F.relu(F.max_pool2d(self.layers.conv1(x), 2))
        #     x = F.relu(F.max_pool2d(self.layers.conv2_drop(self.layers.conv2(x)), 2))
        #     x = x.view(-1, 160)
        #     x = self.layers.fc1(x)
        #     return F.log_softmax(x, dim=1)
        
        def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
            return Network.default_update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index)
            
        def fit(self, *, input_output_pairs=None, dataset=None, loader=None, number_of_epochs=3, batch_size=64, shuffle=True):
            return Network.default_fit(self, input_output_pairs=input_output_pairs, dataset=dataset, loader=loader, number_of_epochs=number_of_epochs, batch_size=batch_size, shuffle=shuffle,)
        
        def test(self, loader, correctness_function=None):
            return Network.default_test(self, loader)
        
    model = SamNet()
    # model = AlexNet().to(device)
    train_dataset, test_dataset, train_loader, test_loader = binary_mnist([9])
    model.fit(loader=train_loader, number_of_epochs=3)
    model.test(loader=test_loader)
    
#%% 
# sample network output
# 
if True:
    from tools.basics import *
    network = model
    for each_index in range(100):
        input_data, correct_output = train_dataset[each_index]
        # train_dataset, test_dataset, train_loader, test_loader
        guess = [ round(each, ndigits=0) for each in to_pure(network.forward(to_tensor([ input_data for each in range(64)]).to(torch.device('cuda:0')) ))[0] ]
        actual = to_pure(correct_output)
        index = max_index(guess)
        # loss = network.loss_function(to_tensor(guess).type(torch.float), to_tensor(actual).type(torch.float))
        # print(f"guess: {guess},\t  index: {index},\t actual: {actual}, loss {loss}")
        print(f"guess: {guess},\t  index: {index},\t actual: {actual}")

#%% 
if True:
    
    # TOOD:
    #    no forgetting / not overspealized, regularizaitons possibly compare to dropout, improved transfer learning
    #    interesting/promising results
    #    get pseduocode -- very high level little bit of flexibility, but make sure they're confident they can implement it

    # import json
    # from os.path import join, dirname
    # with open(join(dirname(__file__), 'data.json'), 'w') as outfile:
    #     json.dump(to_pure(results), outfile)

    # __file__ = '/home/jeffhykin/repos/reinforcement_playground/main/agents/informed_vae/mnist_test.py'
    # import json
    # from os.path import join, dirname
    # with open(join(dirname(__file__), 'data.json'), 'r') as in_file:
    #     results = json.load(in_file)
        
    for each_digit in results:
        pass
    each_digit = results[0]
    import silver_spectacle as ss
    ss.display("chartjs", {
        "type": 'line',
        "data": {
            "datasets": [
                {
                    "label": 'split_train',
                    "data": [ sum(each) for each in each_digit['split_train']],
                    "fill": True,
                    "tension": 0.1,
                    "borderColor": 'rgb(75, 192, 192)',
                },
                {
                    "label": 'classifier_train',
                    "data": each_digit['classifier_train'],
                    "fill": True,
                    "tension": 0.1,
                    "borderColor": 'rgb(0, 292, 192)',
                },
                {
                    "label": 'fresh_classifier_train',
                    "data": each_digit['fresh_classifier_train'],
                    "fill": True,
                    "tension": 0.1,
                    "borderColor": 'rgb(0, 92, 192)',
                },
            ]
        },
        "options": {
            "pointRadius": 3,
            "scales": {
                "y": {
                    "min": 0,
                    "max": 0.5,
                }
            }
        }
    })
    # 
    # importance values
    # 
    # latent_spaces_for_training  = to_tensor( torch.from_numpy(b.encoder(train_dataset[index][0]).cpu().detach().numpy()) for index in range(len(train_dataset)) if index < 10)
    # latent_spaces_for_testing   = to_tensor( torch.from_numpy(b.encoder(test_dataset[index][0]).cpu().detach().numpy()) for index in range(len(test_dataset)) if index < 1)

    # import shap

    # model = nn.Sequential(b.decoder, nn.Flatten())
    # explainer = shap.DeepExplainer(model, latent_spaces_for_training)
    # shap_values = explainer.shap_values(latent_spaces_for_testing)


    # import numpy
    # import functools
    # # sum these up elementwise
    # summed = numpy.squeeze(functools.reduce(
    #     lambda each_new, existing: numpy.add(each_new, existing),
    #     # take the absolute value because we just want impactful values, not just neg/pos correlated ones
    #     numpy.abs(shap_values),
    #     numpy.zeros_like(shap_values[0]),
    # ))

    # print('summed = ', summed)

# %%
