import torch
import torch.nn as nn
import numpy as np

from trivial_torch_tools import Sequential, init, convert_each_arg, product

class VAE(nn.Module):
    @init.forward_sequential_method
    @init.save_and_load_methods(model_attributes=["encoder", "decoder"], basic_attributes=["input_shape", "latent_size"])
    def __init__(self, input_shape, z_dim=512,):
        super().__init__()

        self.input_shape = input_shape
        self.z_dim = z_dim
        
        h = tf.layers.conv2d(self.input_tensor,32,4,strides=2,activation=tf.nn.relu,name="enc_conv1",)
        h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
        h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
        h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")

        # encoder
        self.encoder = nn.Sequential(input_shape=input_shape)
        size = 32
        self.encoder.add_module(nn.Conv2d(4, 32, 4, stride=2, padding=1))
        self.encoder.add_module(nn.BatchNorm2d(32))
        self.encoder.add_module(nn.LeakyReLU())
        self.encoder.add_module(nn.Conv2d(32, 64, 4, stride=2, padding=1))
        self.encoder.add_module(nn.BatchNorm2d(64))
        self.encoder.add_module(nn.LeakyReLU())
        self.encoder.add_module(nn.Conv2d(64, 128, 4, stride=2, padding=1))
        self.encoder.add_module(nn.BatchNorm2d(64))
        self.encoder.add_module(nn.LeakyReLU())
        self.encoder.add_module(nn.Conv2d(64, 256, 4, stride=2, padding=1))
        self.encoder.add_module(nn.BatchNorm2d(64))
        self.encoder.add_module(nn.LeakyReLU())
        
        self.conv_out_size = self._get_conv_out_size(input_shape)
        self.mu = nn.Sequential(
            nn.Linear(self.conv_out_size, z_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.log_var = nn.Sequential(
            nn.Linear(self.conv_out_size, z_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        # decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(z_dim, self.conv_out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        # self.decoder_conv = nn.Sequential(
            # nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),
            # nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),
            # nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(),
            # nn.ConvTranspose2d(32, 3, 2, stride=2, padding=0),
            # nn.Sigmoid()
        # )
        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),
            nn.Sigmoid()
        )