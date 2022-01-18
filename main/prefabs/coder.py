from tools.all_tools import *

from tools.basics import product
from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, Sequential, tensor_to_image

class Encoder(nn.Module):
    @init.hardware
    def __init__(self, **config):
        super(Encoder, self).__init__()
        # 
        # options
        # 
        self.input_shape     = config.get('input_shape'    , (1, 28, 28))
        self.output_shape    = config.get('output_shape'   , (10,))
        
        # 
        # layers
        # 
        self.layers = Sequential(input_shape=self.input_shape)
        # 1 input image, 10 output channels, 5x5 square convolution kernel
        self.layers.add_module('conv1',            nn.Conv2d(self.input_shape[0], 10, kernel_size=5))
        self.layers.add_module('conv1_pool',       nn.MaxPool2d(2))
        self.layers.add_module('conv1_activation', nn.ReLU())
        self.layers.add_module('conv2',            nn.Conv2d(10, 10, kernel_size=5))
        self.layers.add_module('conv2_drop',       nn.Dropout2d())
        self.layers.add_module('conv2_pool',       nn.MaxPool2d(2))
        self.layers.add_module('conv2_activation', nn.ReLU())
        self.layers.add_module('flatten',          nn.Flatten(1)) # 1 => skip the first dimension because thats the batch dimension
        self.layers.add_module('fc1',              nn.Linear(self.layers.output_size, product(self.output_shape)))
        self.layers.add_module('fc1_activation',   nn.ReLU())
    
    @forward.to_tensor
    @forward.to_device
    def loss_function(self, model_output, ideal_output):
        return nn.functional.nll_loss(model_output, ideal_output)
    
    @forward.to_tensor
    @forward.to_device
    def forward(self, batch_of_inputs):
        return self.layers.forward(batch_of_inputs)
    
class Decoder(nn.Module):
    @init.hardware
    def __init__(self, **config):
        super(Decoder, self).__init__()
        self.input_shape  = config.get("input_shape"    , (10,))
        self.output_shape = config.get("output_shape"   , (1, 28, 28))
        
        channel_count, *image_shape, = self.output_shape
        
        # 
        # calculate layer sizing
        # 
        conv1_kernel_size = 5
        conv2_channel_size = 10
        conv2_kernel_size = 5
        # element-wise subtraction because kernels add some size
        conv1_image_shape = to_tensor(image_shape) - ((conv2_kernel_size-1) + (conv1_kernel_size-1))
        conv1_image_shape_as_ints_cause_pytorch_is_really_picky = [ int(each) for each in conv1_image_shape ]
        conv1_shape = [ conv2_size*channel_count, *conv1_image_shape_as_ints_cause_pytorch_is_really_picky ]
        
        # 
        # layers
        #
        self.layers = Sequential(input_shape=self.input_shape)
        self.layers.add_module("fn1",            nn.Linear(self.layers.output_size, 400))
        self.layers.add_module("fn1_activation", nn.ReLU(True))
        self.layers.add_module("fn2",             nn.Linear(self.layers.output_size, product(conv1_shape)))
        self.layers.add_module("fn2_activation",  nn.ReLU(True))
        self.layers.add_module("conv1_prep",      nn.Unflatten(1, conv1_shape))
        self.layers.add_module("conv1",           nn.ConvTranspose2d(conv1_shape[0], conv2_channel_size, kernel_size=conv1_kernel_size))
        self.layers.add_module("conv2",           nn.ConvTranspose2d(conv2_channel_size, channel_count, kernel_size=conv2_kernel_size))
        self.layers.add_module("conv2_activation",nn.Sigmoid())
        
        # loss
        self.loss_function = nn.mse_loss()
    
    @forward.to_tensor
    @forward.to_device
    def loss_function(self, model_output, ideal_output):
        # convert from one-hot into number, and send tensor to device
        return nn.functional.mse_loss(model_output, ideal_output)
    
    @forward.to_tensor
    @forward.to_device
    def forward(self, batch_of_inputs):
        return self.layers.forward(batch_of_inputs)
    
class ImageCoder(nn.Module):
    @init.hardware
    def __init__(self, **config):
        super(ImageCoder, self).__init__()
        # 
        # options
        # 
        self.input_shape     = config.get('input_shape'    , (1, 28, 28))
        self.latent_shape    = config.get('latent_shape'   , (30,))
        self.output_shape    = config.get('output_shape'   , self.input_shape)
        self.learning_rate   = config.get('learning_rate'  , 0.01)
        self.momentum        = config.get('momentum'       , 0.5 )
        
        # 
        # layers
        # 
        self.layers = Sequential(input_shape=self.input_shape)
        self.layers.add_module('encoder', Encoder(input_shape=self.input_shape, output_shape=self.latent_shape))
        self.layers.add_module('decoder', Decoder(input_shape=self.latent_shape, output_shape=self.output_shape))
        
        # optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
    
    @forward.to_tensor
    @forward.to_device
    def loss_function(self, model_output, ideal_output):
        ideal_output = opencv_image_to_torch_image(ideal_output)
        return F.mse_loss(model_output, ideal_output)
    
    @forward.to_tensor
    @forward.to_device
    def encode(self, input_data):
        return self.layers.encoder.forward(input_data)
    
    @forward.to_tensor
    @forward.to_device
    @forward.to_batched_tensor(number_of_dimensions=4) # batch_size, color_channels, image_width, image_height
    @forward.from_opencv_image_to_torch_image
    def forward(self, input_batch):
        return self.layers.forward(input_batch)
    
    def update_weights(self, batch_of_inputs, batch_of_ideal_outputs, epoch_index, batch_index):
        self.optimizer.zero_grad()
        batch_of_actual_outputs = self.forward(batch_of_inputs)
        loss = self.loss_function(batch_of_actual_outputs, batch_of_ideal_outputs)
        loss.backward()
        self.optimizer.step()
        return loss


import torchvision
import torch
import silver_spectacle as ss
from tools.pytorch_tools import opencv_image_to_torch_image, read_image, torch_image_to_opencv_image, to_tensor, init, forward, Sequential, tensor_to_image
randomize_image = Sequential(
    torchvision.transforms.Pad(5, fill=0, padding_mode='constant'),
    torchvision.transforms.RandomPerspective(distortion_scale=0.03, p=1, fill=0),
    torchvision.transforms.ColorJitter(hue=.03, saturation=.03),
    torchvision.transforms.RandomRotation(1),
)
img = read_image("img.ignore.png")[:3,:,:,] # remove the alpha channel 
img_bumped = randomize_image(img)
ss.DisplayCard("quickImage", torch_image_to_opencv_image(img))
ss.DisplayCard("quickImage", torch_image_to_opencv_image(randomize_image(img)))

from super_map import LazyDict
class ImageQueBump(nn.Module):
    @init.hardware
    def __init__(self, **config):
        super(ImageQueBump, self).__init__()
    
    def forward(self, several_images):
        parameters = LazyDict(
            padding_amount=0.03 * max(several_images.shape),
            rotation=LazyDict(
                max=5,
                min=-5,
                standard_deviation=3,
            ),
            translation=LazyDict(
                max=12,
                min=-12,
                standard_deviation=5,
            ),
            perspective=LazyDict(
                horizontal_shift=LazyDict(
                    max=10, # percent
                    min=-10,
                ),
                vertical_shift=LazyDict(
                    max=10, # percent
                    min=-10,
                )
            )
        )
        import torchvision.transforms.functional as F
        for each_img in several_images:
            F.rotate(
                angle=5,
                img=F.perspective(
                    startpoints=[
                        [x,y], # top-left
                        [x,y], # top-right
                        [x,y], # bottom-right
                        [x,y], # bottom-left
                    ],
                    endpoints=[
                        [x,y], # top-left
                        [x,y], # top-right
                        [x,y], # bottom-right
                        [x,y], # bottom-left
                    ],
                    img=F.pad(
                        padding=padding_amount,
                        fill=0,
                        padding_mode='constant',
                        img=F.adjust_hue(
                            hue_factor=0.05,
                            img=each_img,
                        ),
                    ),
                )
            )
            
        return self.layers.forward(batch_of_inputs)
    
