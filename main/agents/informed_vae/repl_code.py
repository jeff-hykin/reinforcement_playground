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

    torch.manual_seed(1)

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
            
            return torch.stack(reshaped_list)    
                
    def onehot_argmax(tensor):
        tensor = to_tensor(tensor)
        the_max = max(each for each in tensor)
        onehot_tensor = torch.zeros_like(tensor)
        for each_index, each_value in enumerate(tensor):
            if each_value == the_max:
                onehot_tensor[each_index] = 1
        return onehot_tensor

    def batch_input_and_output(inputs, outputs, batch_size):
        from tools.basics import bundle
        batches = zip(bundle(inputs, batch_size), bundle(outputs, batch_size))
        for each_input_batch, each_output_batch in batches:
            yield to_tensor(each_input_batch), to_tensor(each_output_batch)

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

    def autoencodeify(dataset):
        class AutoDataset(dataset):
            def __init__(self, *args, **kwargs):
                super(AutoDataset, self).__init__(*args, **kwargs)
            
            def __getitem__(self, index):
                an_input, corrisponding_output = super(AutoDataset, self).__getitem__(index)
                return an_input, an_input

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
        # test_dataset = Dataset(**{**options, "train":False})
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=lambda *args:range(len(train_dataset))),
            batch_size=64,
            # shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            # sampler=ImbalancedDatasetSampler(test_dataset),
            batch_size=1000,
            shuffle=True,
        )
        return train_dataset, test_dataset, train_loader, test_loader

#%% 
# informed_vae/main.py
# 
if True:
    # 
    # ImageEncoder
    # 
    class ImageEncoder(ImageModelSequential):
        '''
        examples:
            an_encoder = ImageEncoder()
            from tools.all_tools import *
            # img is just a torch tensor
            img = read_image(mnist_dataset.path+"/img_0/data.jpg")
            an_encoder.forward(img)
        notes:
            an external network is going to be the one updating the gradients
            traditionally it would be the decoder, figuring out the best way to decode
            however it can also be the core_agent, figuring out what features would help with its decision process
            Ideally it will be both those things combined, or something more advanced
        '''
        def __init__(self, input_shape=(1, 28, 28), latent_shape=(10,), loss_function=None, **config):
            # this statement is a helper from ImageModelSequential
            with self.setup(input_shape=input_shape, output_shape=latent_shape, loss_function=loss_function):
                # gives us access to
                #     self.print()
                #     self.input_feature_count
                #     self.output_feature_count
                #     self.layers
                #     self.loss()
                #     self.gradients
                #     self.update_gradients()  # using self.loss
                
                # 
                # Layers
                # 
                self.layers.add_module("layer1", nn.Linear(self.size_of_last_layer, int(self.input_feature_count/2)))
                self.layers.add_module("layer1_activation", nn.ReLU())
                self.layers.add_module("layer2", nn.Linear(self.size_of_last_layer, 64))
                self.layers.add_module("layer2_activation", nn.ReLU())
                self.layers.add_module("layer3", nn.Linear(self.size_of_last_layer, self.output_feature_count))
                self.layers.add_module("layer3_activation", nn.Sigmoid())
                
                # default to squared error loss
                def loss_function(input_batch, ideal_output_batch):
                    actual_output_batch = self.forward(input_batch).to(self.device)
                    ideal_output_batch = ideal_output_batch.to(self.device)
                    return torch.mean((actual_output_batch - ideal_output_batch)**2)
                    
                self.loss_function = loss_function
        
        def update_weights(self, input_batch, ideal_outputs_batch, **config):
            # 
            # data used inside the update
            # 
            step_size = config.get("step_size", 0.000001)
            gradients = self.compute_gradients_for(
                input_batch=input_batch,
                ideal_outputs_batch=ideal_outputs_batch,
                loss_function=self.loss_function
            )
            
            # 
            # the actual update
            # 
            # turn off gradient tracking because things are about to be updated
            with torch.no_grad():
                for gradients, each_layer in zip(gradients, self.weighted_layers):
                    weight_gradient, bias_gradient = gradients
                    
                    print('each_layer.weight = ', each_layer.weight)
                    print('step_size * weight_gradient = ', step_size * weight_gradient)
                    each_layer.weight += step_size * weight_gradient
                    each_layer.bias   += step_size * bias_gradient
            
            
            # turn gradient-tracking back on
            for each in self.layers:
                each.requires_grad = True
        
        def fit(self, input_output_pairs, **config):
            batch_size     = config.get("batch_size"   , 32)
            epochs         = config.get("epochs"       , 10)
            update_options = config.get("update_options", {}) # step_size can be in here
            
            # convert so that input_batch is a single tensor and output_batch is a single tensor
            all_inputs  = (each for each, _ in input_output_pairs)
            all_outputs = (each for _   , each in input_output_pairs)
            
            from tools.pytorch_tools import batch_input_and_output
            batch_number = 0
            losses = []
            for each_epoch in range(epochs):
                for batch_of_inputs, batch_of_ideal_outputs in batch_input_and_output(all_inputs, all_outputs, batch_size):
                    batch_number += 1
                    print('batch_number = ', batch_number)
                    losses.append(self.update_weights(batch_of_inputs, batch_of_ideal_outputs, **update_options))
            
            return self

    from tools.dataset_tools import mnist_dataset
    def test_encoder():
        from tools.dataset_tools import mnist_dataset
        from tools.pytorch_tools import read_image
        
        # 
        # forward pass
        # 
        dummy_encoder = ImageEncoder()
        # grab the first Mnist image
        img = read_image(mnist_dataset.path+"/img_0/data.jpg")
        encoded_output = dummy_encoder.forward(img)
        print('encoded_output = ', encoded_output)
        
        # 
        # training
        # 
        return dummy_encoder.fit(
            # mnist_dataset is an iterable with each element being an input output pair
            input_output_pairs=mnist_dataset,
        )

#%% 
# informed_vae/wip.py
# 
if True:
    class SGD(Optimizer):
        r"""Implements stochastic gradient descent (optionally with momentum).

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float): learning rate
            momentum (float, optional): momentum factor (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            dampening (float, optional): dampening for momentum (default: 0)

        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()

        __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

        .. note::
            The implementation of SGD with Momentum subtly differs from
            Sutskever et. al. and implementations in some other frameworks.

            Considering the specific case of Momentum, the update can be written as

            .. math::
                \begin{aligned}
                    v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                    p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
                \end{aligned}

            where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
            parameters, gradient, velocity, and momentum respectively.

            This is in contrast to Sutskever et. al. and
            other frameworks which employ an update of the form

            .. math::
                \begin{aligned}
                    v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                    p_{t+1} & = p_{t} - v_{t+1}.
                \end{aligned}

        """

        @classmethod
        def sgd(
            cls,
            parameters,
            gradients,
            momentum_buffer_list,
            *,
            weight_decay: float,
            momentum: float,
            lr: float,
            dampening: float,
        ):
            r"""
            Functional API that performs SGD algorithm computation.
            See :class:`~torch.optim.SGD` for details.
            """
            for index, parameters in enumerate(parameters):

                gradient = gradients[index]
                if weight_decay != 0:
                    gradient = gradient.add(parameters, alpha=weight_decay)

                if momentum != 0:
                    buffer = momentum_buffer_list[index]

                    if buffer is None:
                        buffer = torch.clone(gradient).detach()
                        momentum_buffer_list[index] = buffer
                    else:
                        buffer.mul_(momentum).add_(gradient, alpha=1 - dampening)

                    gradient = buffer

                parameters.add_(gradient, alpha=-lr)
        
        def __init__(self, model_parameters, lr=required, momentum=0, dampening=0, weight_decay=0):
            if lr is not required and lr < 0.0:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if momentum < 0.0:
                raise ValueError("Invalid momentum value: {}".format(momentum))
            if weight_decay < 0.0:
                raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

            defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
            super(SGD, self).__init__(model_parameters, defaults)

        def __setstate__(self, state):
            super(SGD, self).__setstate__(state)

        @torch.no_grad()
        def step(self, closure=None):
            """Performs a single optimization step.

            Args:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                params_with_grad = []
                gradients = []
                momentum_buffer_list = []
                weight_decay = group['weight_decay']
                momentum     = group['momentum']
                dampening    = group['dampening']
                lr           = group['lr']

                for parameter in group['params']:
                    if parameter.grad is not None:
                        params_with_grad.append(parameter)
                        gradients.append(parameter.grad)

                        state = self.state[parameter]
                        if 'momentum_buffer' not in state:
                            momentum_buffer_list.append(None)
                        else:
                            momentum_buffer_list.append(state['momentum_buffer'])
                
                self.sgd(
                    params_with_grad,
                    gradients,
                    momentum_buffer_list,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    lr=lr,
                    dampening=dampening,
                )

                # update momentum_buffers in state
                for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum_buffer

            return loss

    class ImageEncoder(ImageModelSequential):
        def __init__(self, **config):
            self.input_shape   = config.get("input_shape", (1, 28, 28))
            self.output_shape  = config.get("output_shape", (10,))
            self.learning_rate = config.get("learning_rate", 0.01)
            self.momentum      = config.get("momentum", 0.5)
            self.log_interval  = config.get("log_interval", 10)
            
            with self.setup(input_shape=self.input_shape, output_shape=self.output_shape):
                self.layers.add_module("conv1", nn.Conv2d(1, 10, kernel_size=5))
                self.layers.add_module("conv1_pool", nn.MaxPool2d(2))
                self.layers.add_module("conv1_activation", nn.ReLU())
                
                self.layers.add_module("conv2", nn.Conv2d(10, 20, kernel_size=5))
                self.layers.add_module("conv2_dropout", nn.Dropout2d())
                self.layers.add_module("conv2_pool", nn.MaxPool2d(2))
                self.layers.add_module("conv2_activation", nn.ReLU())
                
                self.layers.add_module("flatten", nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
                self.layers.add_module("fc1", nn.Linear(self.size_of_last_layer, 50))
                self.layers.add_module("fc1_activation", nn.ReLU())
                self.layers.add_module("fc1_dropout", nn.Dropout2d())
                
                self.layers.add_module("fc2", nn.Linear(self.size_of_last_layer, product(self.output_shape)))
                self.layers.add_module("fc2_activation", nn.LogSoftmax(dim=-1))
            
            def NLLLoss(batch_of_actual_outputs, batch_of_ideal_outputs):
                output = batch_of_actual_outputs[range(len(batch_of_ideal_outputs)), batch_of_ideal_outputs]
                return -output.sum()/len(output)
            
            self.loss_function = NLLLoss
            self.optimizer = SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        def train_and_test_on_mnist(self):
            from tools.basics import temp_folder
            # 
            # training dataset
            # 
            train_loader = torch.utils.data.DataLoader(
                autoencodeify(torchvision.datasets.MNIST)(
                    f"{temp_folder}/files/",
                    train=True,
                    download=True,
                    transform=torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                ),
                batch_size=64,
                shuffle=True,
            )

            # 
            # testing dataset
            # 
            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                    f"{temp_folder}/files/",
                    train=False,
                    download=True,
                    transform=torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                ),
                batch_size=1000,
                shuffle=True,
            )
            
            self.test(test_loader)
            self.fit(loader=train_loader, number_of_epochs=3)
            self.test(test_loader)
            
            return self
            
        
    class ImageDecoder(ImageModelSequential):
        def __init__(self, **config):
            self.input_shape   = config.get("input_shape", (10,))
            self.output_shape  = config.get("output_shape", (1, 28, 28))
            self.learning_rate = config.get("learning_rate", 0.01)
            self.momentum      = config.get("momentum", 0.5)
            self.log_interval  = config.get("log_interval", 10)
            
            with self.setup(input_shape=self.input_shape, output_shape=self.output_shape):
                self.layers.add_module("fn1", nn.Linear(self.size_of_last_layer, 400))
                self.layers.add_module("fn1_activation", nn.ReLU(True))
                
                self.layers.add_module("fn2", nn.Linear(self.size_of_last_layer, 4000))
                self.layers.add_module("fn2_activation", nn.ReLU(True))
                
                conv1_shape = [ 10, 20, 20 ] # needs to mupltiply together to be the size of the previous layer (currently 4000)
                conv2_size = 10
                self.layers.add_module("conv1_prep", nn.Unflatten(1, conv1_shape))
                self.layers.add_module("conv1", nn.ConvTranspose2d(conv1_shape[0], conv2_size, kernel_size=5))
                self.layers.add_module("conv2", nn.ConvTranspose2d(conv2_size, 1, kernel_size=5))
                self.layers.add_module("conv2_activation", nn.Sigmoid())
            
                self.loss_function = nn.MSELoss()
                self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
    class ImageAutoEncoder(ImageModelSequential):
        def __init__(self, **config):
            self.input_shape   = config.get("input_shape", (1, 28, 28))
            self.latent_shape  = config.get("latent_shape", (10,))
            self.output_shape  = config.get("output_shape", (1, 28, 28))
            self.learning_rate = config.get("learning_rate", 0.01)
            self.momentum      = config.get("momentum", 0.5)
            self.log_interval  = config.get("log_interval", 10)
            
            with self.setup(input_shape=self.input_shape, output_shape=self.output_shape):
                # 
                # encoder
                # 
                self.encoder = ImageEncoder(
                    input_shape=self.input_shape,
                    output_shape=self.latent_shape,
                )
                self.layers.add_module("encoder", self.encoder)
                # 
                # decoder
                # 
                self.decoder = ImageDecoder(
                    input_shape=self.latent_shape,
                    output_shape=self.output_shape,
                )
                self.layers.add_module("decoder", self.decoder)
                
            self.loss_function = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
            self.optimizer.zero_grad()
            batch_of_actual_outputs = self.forward(batch_of_inputs)
            loss = self.loss_function(batch_of_actual_outputs, batch_of_inputs)
            loss.backward()
            self.optimizer.step()
            return loss
        
        # TODO: test(self) needs to be changed, but its a bit difficult to make it useful
        
        def train_and_test_on_mnist(self):
            # 
            # modify Mnist so that the input and output are both the image
            # 
            from tools.basics import temp_folder
            train_loader = torch.utils.data.DataLoader(
                autoencodeify(torchvision.datasets.MNIST)(
                    f"{temp_folder}/files/",
                    train=True,
                    download=True,
                    transform=torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                ),
                batch_size=64,
                shuffle=True,
            )
            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                    f"{temp_folder}/files/",
                    train=False,
                    download=True,
                    transform=torchvision.transforms.Compose(
                        [
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                ),
                batch_size=1000,
                shuffle=True,
            )
            
            # FIXME: add a testing method (probably related to confusion_matrix) for auto-encoder
            # self.test(test_loader)
            # TODO: autoencodeify the train loader inside the fit function
            self.fit(loader=train_loader, number_of_epochs=3)
            # self.test(test_loader)
            
        
        def generate_confusion_matrix(self, test_loader):
            from tools.basics import product
            number_of_outputs = product(self.latent_shape)
            confusion_matrix = torch.zeros(number_of_outputs, number_of_outputs)
            test_losses = []
            test_loss = 0
            correct = 0
            
            self.eval()
            with torch.no_grad():
                for batch_of_inputs, batch_of_ideal_outputs in test_loader:
                    latent_space_activation_batch = self.encoder.forward(batch_of_inputs)
                    for each_activation_space, each_ideal_output in zip(latent_space_activation_batch, batch_of_ideal_outputs):
                        # which index was chosen
                        predicted_index = numpy.argmax(each_activation_space)
                        actual_index    = numpy.argmax(each_ideal_output)
                        confusion_matrix[actual_index][predicted_index] += 1
            
            return confusion_matrix
        
        def importance_identification(self, train_dataset, test_dataset, training_size=10, testing_size=1):
            """
            Outputs:
                importance_values, shap_values
                
                the shap_values
                    is A list
                    The length of list is the size of the model's output (one element per output parameter)
                    Every element in the list is a numpy array
                    Every element has the shape of the model's input
                    A value of 0 means no correlation
                    negative values means negatively correlated (similar for positive values)
                
                the importance values
                    is a numpy array
                    The shape is the same shape as the input for the model
                    The values are the relative importance of each latent parameter
                    All values are positive
                    Larger values = more important
            """
            import shap
            
            # TODO: improve me, these values are converted to and from numpy values basically as a means of copying to shead information such as gradient tracking info
            # note: the inputs are all encoded because we're trying to explain the latent/encoded space
            latent_spaces_for_training  = to_tensor( torch.from_numpy(b.encoder(train_dataset[index][0]).cpu().detach().numpy()) for index in range(len(train_dataset)) if index < training_size)
            latent_spaces_for_testing   = to_tensor( torch.from_numpy(b.encoder(test_dataset[index][0] ).cpu().detach().numpy()) for index in range(len(test_dataset )) if index < testing_size)
            
            # DeepExplainer needs the output to be flat for some reason
            # use only the decoder explains the latent space instead of going back to the image
            model = nn.Sequential(
                self.decoder,
                nn.Flatten(),
            )
            explainer = shap.DeepExplainer(model, latent_spaces_for_training)
            shap_values = explainer.shap_values(latent_spaces_for_testing)
            
            import numpy
            import functools
            # sum these up elementwise
            summed = numpy.squeeze(functools.reduce(
                lambda each_new, existing: numpy.add(each_new, existing),
                # take the absolute value because we just want impactful values, not just neg/pos correlated ones
                numpy.abs(shap_values),
                numpy.zeros_like(shap_values[0]),
            ))
            return summed, shap_values


    class ImageClassifier(ImageAutoEncoder):
        def __init__(self, **config):
            self.input_shape   = config.get("input_shape", (1, 28, 28))
            self.latent_shape  = config.get("latent_shape", (20,))
            self.output_shape  = config.get("output_shape", (2,))
            self.learning_rate = config.get("learning_rate", 0.01)
            self.momentum      = config.get("momentum", 0.5)
            self.log_interval  = config.get("log_interval", 10)
            
            with self.setup(input_shape=self.input_shape, output_shape=self.output_shape):
                # 
                # encoder
                # 
                self.encoder       = config.get("encoder", ImageEncoder(
                    input_shape=self.input_shape,
                    output_shape=self.latent_shape,
                ))
                self.layers.add_module("encoder", self.encoder)
                # 
                # task (classifier)
                # 
                self.task_network = nn.Sequential(
                    nn.Linear(product(self.latent_shape), 2), # binary classification
                    nn.Sigmoid(),
                )
                self.layers.add_module("task_network", self.task_network)
                
            self.classifier_loss_function = self.loss_function = nn.BCELoss()
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
            # set the gradient values to zero before accumulating them
            self.zero_grad()
            batch_of_latent_vectors = self.encoder.forward(batch_of_inputs)
            batch_of_classified_images = self.task_network.forward(batch_of_latent_vectors)
            task_loss = self.classifier_loss_function(batch_of_classified_images.type(torch.float), batch_of_ideal_outputs.type(torch.float))
            task_loss.backward()
            
            # call step after both losses have propogated backward
            self.optimizer.step()
            return task_loss

    class ImageClassifier2(ImageAutoEncoder):
        def __init__(self, **config):
            self.input_shape         = config.get("input_shape"        , (1 , 28, 28))
            self.latent_shape        = config.get("latent_shape"       , (20, ))
            self.output_shape        = config.get("output_shape"       , (2 , ))
            self.decoded_shape       = config.get("decoded_shape"      , (1 , 28, 28))
            self.learning_rate       = config.get("learning_rate"      , 0.01)
            self.momentum            = config.get("momentum"           , 0.5)
            self.log_interval        = config.get("log_interval"       , 10)
            self.decoding_importance = config.get("decoding_importance", 0)
            
            with self.setup(input_shape=self.input_shape, output_shape=self.output_shape):
                # 
                # encoder
                # 
                self.encoder       = config.get("encoder", ImageEncoder(
                    input_shape=self.input_shape,
                    output_shape=self.latent_shape,
                ))
                self.layers.add_module("encoder", self.encoder)
                # 
                # task (classifier)
                # 
                self.task_network = nn.Sequential(
                    nn.Linear(product(self.latent_shape), 2), # binary classification
                    nn.Sigmoid(),
                )
                self.layers.add_module("task_network", self.task_network)
                # 
                # task (decoder)
                # 
                # self.decoder       = config.get("decoder", ImageDecoder(
                #     input_shape=self.latent_shape,
                #     output_shape=self.decoded_shape,
                # ))
                
                
            self.decoder_loss_function = nn.MSELoss()
            self.classifier_loss_function = self.loss_function = nn.BCELoss()
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
            # set the gradient values to zero before accumulating them
            self.zero_grad()
            batch_of_latent_vectors = self.encoder.forward(batch_of_inputs)
            # loss #1
            # batch_of_decoded_images = self.decoder.forward(batch_of_latent_vectors)
            # decoder_loss = self.decoder_loss_function(batch_of_decoded_images, batch_of_inputs) * self.decoding_importance
            # decoder_loss.backward(retain_graph=True)
            # loss #2 
            # FIXME: figure out how to make this loss relatively more important (maybe a scalar would work?)
            batch_of_classified_images = self.task_network.forward(batch_of_latent_vectors)
            task_loss = self.classifier_loss_function(batch_of_classified_images.type(torch.float), batch_of_ideal_outputs.type(torch.float)) 
            task_loss.backward()
            
            # call step after both losses have propogated backward
            self.optimizer.step()
            return task_loss
            
    class SplitAutoEncoder(ImageAutoEncoder):
        def __init__(self, **config):
            self.input_shape   = config.get("input_shape", (1, 28, 28))
            self.latent_shape  = config.get("latent_shape", (20,))
            self.output_shape  = config.get("output_shape", (2,))
            self.decoded_shape = config.get("decoded_shape", (1, 28, 28))
            self.learning_rate = config.get("learning_rate", 0.01)
            self.momentum      = config.get("momentum", 0.5)
            self.log_interval  = config.get("log_interval", 10)
            self.decoding_importance = config.get("decoding_importance", 1)
            
            with self.setup(input_shape=self.input_shape, output_shape=self.output_shape):
                # 
                # encoder
                # 
                self.encoder       = config.get("encoder", ImageEncoder(
                    input_shape=self.input_shape,
                    output_shape=self.latent_shape,
                ))
                self.layers.add_module("encoder", self.encoder)
                # 
                # task (classifier)
                # 
                self.task_network = nn.Sequential(
                    nn.Linear(product(self.latent_shape), 2), # binary classification
                    nn.Sigmoid(),
                )
                self.layers.add_module("task_network", self.task_network)
                # 
                # task (decoder)
                # 
                # self.decoder       = config.get("decoder", ImageDecoder(
                #     input_shape=self.latent_shape,
                #     output_shape=self.decoded_shape,
                # ))
                
                
            self.decoder_loss_function = nn.MSELoss()
            self.classifier_loss_function = self.loss_function = nn.BCELoss()
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
            # set the gradient values to zero before accumulating them
            self.zero_grad()
            batch_of_latent_vectors = self.encoder.forward(batch_of_inputs)
            # loss #1
            # batch_of_decoded_images = self.decoder.forward(batch_of_latent_vectors)
            # decoder_loss = self.decoder_loss_function(batch_of_decoded_images, batch_of_inputs) * self.decoding_importance
            # decoder_loss.backward(retain_graph=True)
            # loss #2 
            # FIXME: figure out how to make this loss relatively more important (maybe a scalar would work?)
            batch_of_classified_images = self.task_network.forward(batch_of_latent_vectors)
            task_loss = self.classifier_loss_function(batch_of_classified_images.type(torch.float), batch_of_ideal_outputs.type(torch.float)) 
            task_loss.backward()
            
            # call step after both losses have propogated backward
            self.optimizer.step()
            return task_loss    
        
#%% 
if True:
    __file__ = '/home/jeffhykin/repos/reinforcement_playground/main/agents/informed_vae/mnist_test.py'
    
#%% 
# test_split_network
# 
if True:


    # 
    # train and test
    # 
    split = SplitAutoEncoder()
    classifier = ImageClassifier()
    classifier2 = ImageClassifier2()
    for each in [9]:
        result = {}
        
        # 
        # split
        # 
        train_dataset, test_dataset, train_loader, test_loader = binary_mnist([each])
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
        # classifier2
        # 
        train_dataset, test_dataset, train_loader, test_loader = binary_mnist([each])
        # reset the task network part (last few layers)
        classifier2.task_network = nn.Sequential(
            nn.Linear(product(classifier2.latent_shape), 2), # binary classification
            nn.Sigmoid(),
        )
        result["classifier2_train"] = classifier2.fit(loader=train_loader, number_of_epochs=3)
        result["classifier2_test"] = classifier2.test(test_loader)
        
        # 
        # fresh_classifier
        # 
        # fresh_classifier = ImageClassifier()
        # train_dataset, test_dataset, train_loader, test_loader = binary_mnist([each])
        # result["fresh_classifier_train"] = fresh_classifier.fit(loader=train_loader, number_of_epochs=3)
        # result["fresh_classifier_test"] = fresh_classifier.test(test_loader)
        
        # save
        results.append(result)

    for each_iter in results:
        result_string = "results: "
        for each_key, each_network_result in each_iter.items():
            if "test" in each_key:
                result_string += f'({each_key.replace("test", "")}: {each_network_result})  '
        print(result_string)

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
