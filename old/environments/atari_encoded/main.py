import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from old.environments.atari.main import *
from old.environments.atari.main import Environment as PureAtartEnvironment
from old.environments.atari_encoded.autoencoder import ImageAutoEncoder


class Environment(PureAtartEnvironment):
    def __init__(self, *args, latent_shape, **kwargs):
        super(Environment, self).__init__(*args, **kwargs)
        self.autoencoder = None
        self.latent_shape = latent_shape
        
    def step(self, action):
        state, reward, is_game_over, debugging_info = super().step(action)
        # init the encoder once the first observation is made
        if self.autoencoder == None:
            self.autoencoder = ImageAutoEncoder(
                input_shape=state.shape,
                latent_shape=self.latent_shape,
            )
            # FIXME: how should the model be loaded or when should it be trained
        
        return self.autoencoder.encode(state), reward, is_game_over, debugging_info