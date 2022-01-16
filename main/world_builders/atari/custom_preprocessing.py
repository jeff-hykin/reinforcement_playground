# file from: https://towardsdatascience.com/deep-q-network-with-pytorch-146bfa939dfe
import torch
import torch.nn as nn
import random
import pickle 
import gym
import numpy as np
import collections 
import cv2
import time

from tools.basics import product
from tools.frame_que import FrameQue
from tools.debug import debug

def preprocess(env, frame_buffer_size, frame_sample_rate):
    
    env.prev_unpreprocessed_frame = None

    class MaxAndSkipEnv(gym.Wrapper):
        """
            Each action of the agent is repeated over skip frames
            return only every `skip`-th frame
        """
        def __init__(self, env=None, skip=4):
            super(MaxAndSkipEnv, self).__init__(env)
            # most recent raw observations (for max pooling across time steps)
            self._obs_buffer = collections.deque(maxlen=2)
            self._skip = skip

        def step(self, action):
            total_reward = 0.0
            done = None
            for _ in range(self._skip):
                obs, reward, done, info = self.env.step(action)
                self._obs_buffer.append(obs)
                total_reward += reward
                if done:
                    break
            max_frame = np.max(np.stack(self._obs_buffer), axis=0)
            
            env.prev_unpreprocessed_frame = max_frame
            
            return max_frame, total_reward, done, info

        def reset(self):
            """Clear past frame buffer and init to first obs"""
            self._obs_buffer.clear()
            obs = self.env.reset()
            self._obs_buffer.append(obs)
            return obs


    class RescaleAndGrayscale(gym.ObservationWrapper):
        """
        Downsamples/Rescales each frame to size 84x84 with greyscale
        """
        def __init__(self, env=None, *, height, width):
            super(RescaleAndGrayscale, self).__init__(env)
            self.height = height
            self.width = width
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width), dtype=np.uint8)

        def observation(self, obs):
            return cv2.cvtColor(
                cv2.resize(
                    obs,
                    (self.height, self.width),
                    interpolation=cv2.INTER_AREA
                ),
                cv2.COLOR_BGR2GRAY
            )

    class ImageToPyTorch(gym.ObservationWrapper):
        """
        Each frame is converted to PyTorch tensors
        """
        def __init__(self, env, low=0, high=255, dtype=np.uint8):
            super(ImageToPyTorch, self).__init__(env)
            old_shape = self.observation_space.shape
            self.observation_space = gym.spaces.Box(
                low=low,
                high=high,
                shape=(old_shape[-1], old_shape[0], old_shape[1]),
                dtype=dtype,
            )

        def observation(self, observation):
            return np.moveaxis(observation, 2, 0)

        
    class BufferWrapper(gym.ObservationWrapper):
        """
        Only every k-th frame is collected by the buffer
        """
        def __init__(self, env, n_steps, dtype=np.uint8):
            super(BufferWrapper, self).__init__(env)
            self.dtype = dtype
            old_space = env.observation_space
            debug.old_space = old_space
            self.observation_space = gym.spaces.Box(
                old_space.low.reshape((1, *old_space.low.shape)).repeat(n_steps, axis=0),
                old_space.high.reshape((1, *old_space.high.shape)).repeat(n_steps, axis=0),
                dtype=dtype,
            )

        def reset(self):
            self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
            return self.observation(self.env.reset())

        def observation(self, observation):
            print('self.buffer.shape = ', self.buffer.shape)
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = observation
            return self.buffer


    class PixelNormalization(gym.ObservationWrapper):
        """
        Normalize pixel values in frame --> 0 to 1
        """
        def observation(self, obs):
            return np.array(obs).astype(np.float32) / 255.0
    
    
    env = MaxAndSkipEnv(env, skip=frame_sample_rate)
    env = RescaleAndGrayscale(env, height=84, width=84)
    # env = ImageToPyTorch(env)
    env = BufferWrapper(env, frame_buffer_size, dtype=np.uint8)
    # env = PixelNormalization(env) # this breaks the baselines agent
    
    return env