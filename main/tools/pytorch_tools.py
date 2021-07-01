import torch
import torch.nn as nn

from tools.basics import product, bundle

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
                self.print = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
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
    
    @property
    def weighted_layers(self):
        return [ each_layer for each_layer in self.layers if hasattr(each_layer, "weight") ]
    
    @property
    def gradients(self):
        gradients = []
        for each_layer in self.weighted_layers:
            # if it has weights (e.g. not an activation function "layer")
            if hasattr(each_layer, "bias"):
                gradients.append((each_layer.weight.grad, each_layer.bias.grad))
            else:
                gradients.append((each_layer.weight.grad, None))
        return gradients
        
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
        if loss_function is None:
            loss_function = self.loss_function
        
        torch.save(self.state_dict(), "./tmp.dont-sync.model")
        
        loss_value = loss_function(input_batch, ideal_outputs_batch)
        # compute the gradients
        loss_value.backward(retain_graph=retain_graph)
        # get the gradients
        gradients = self.gradients
        # restore the previous values 
        self.load_state_dict(torch.load("./tmp.dont-sync.model"), strict=False)
        # return the gradients
        return gradients