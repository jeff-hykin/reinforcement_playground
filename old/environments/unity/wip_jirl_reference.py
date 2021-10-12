# Original author: Roma Sokolkov
# Edited by Antonin Raffin
# Edited by Sheelabhadra Dey

import random
import time
import os
import warnings

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time

MIN_THROTTLE = 0.45
MAX_THROTTLE =  0.6
MAX_STEERING_DIFF = 0.15
JERK_REWARD_WEIGHT = 0.0
MAX_STEERING = 1
MIN_STEERING = - MAX_STEERING
STEERING_GAIN = 1
STEERING_BIAS = 0

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

unity_gym_env = UnityToGymWrapper(
    UnityEnvironment(),
    allow_multiple_obs=True, # not exactly sure what this does,
)

class Envrionment(object):
    def __init__(
        self,
        vae=None,
        min_throttle=0.45,
        max_throttle=0.6,
        n_command_history=0,
        frame_skip=1,
        n_stack=1,
        action_lambda=0.5,
    ):
        # copy args
        self.vae           = vae
        self.min_throttle  = min_throttle
        self.max_throttle  = max_throttle
        self.frame_skip    = frame_skip
        self.action_lambda = action_lambda
        
        # init
        self.throttle = 0
        self.steering = 0
        self.z_size = None if vae is None else vae.z_size
        
        # emulate gym env
        self.observation_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, self.z_size),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-MAX_STEERING, -1]),
            high=np.array([MAX_STEERING, 1]),
            dtype=np.float32,
        )

        # save last n commands
        self.n_commands = 2
        self.n_command_history = n_command_history
        self.command_history = np.zeros((1, self.n_commands * n_command_history))
        self.n_stack = n_stack
        self.stacked_obs = None

    def step(self, action):
        steering, throttle = action
        
        # 
        # process throttle
        # 
        # Convert from [-1, 1] to [0, 1]
        zero_to_one = (throttle + 1) / 2
        # Convert from [0, 1] to [min, max]
        throttle = (1 - zero_to_one) * self.min_throttle + self.max_throttle * zero_to_one
        throttle = float(throttle)
        
        # 
        # process steering angle
        # 
        # Clip steering angle rate to enforce continuity
        if self.n_command_history > 0:
            prev_steering = self.command_history[0, -2]
            max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
            diff = np.clip(steering - prev_steering, -max_diff, max_diff)
            steering = prev_steering + diff
        
        # 
        # apply action
        # 
        self.throttle = throttle
        self.steering = float(steering) * STEERING_GAIN + STEERING_BIAS
        # Repeat action if using frame_skip
        for _ in range(self.frame_skip):
            raw_input, reward, done, debugging_info = unity_gym_env.step(np.array([ self.steering, self.throttle ]))
            camera_input, = raw_input
            observation = self.vae.encode_from_raw_image(camera_input)

        return observation, reward, done, debugging_info

    def reset(self):
        # 
        # reset the unity env
        # 
        camera_input, = unity_gym_env.reset()
        print("ENV: resetting. Camera_input is", camera_input.shape)
        observation = self.vae.encode_from_raw_image(camera_input)
        
        # 
        # reset vars
        # 
        self.throttle = 0
        self.steering = 0
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))
        if self.n_command_history > 0:
            observation = np.concatenate((observation, self.command_history), axis=-1)
        
        return observation
    
    def jerk_penalty(self):
        """
        Add a continuity penalty to limit jerk.
        :return: (float)
        """
        jerk_penalty = 0
        if self.n_command_history > 1:
            # Take only last command into account
            for i in range(1):
                steering = self.command_history[0, -2 * (i + 1)]
                prev_steering = self.command_history[0, -2 * (i + 2)]
                steering_diff = (prev_steering - steering) / (
                    MAX_STEERING - MIN_STEERING
                )

                if abs(steering_diff) > MAX_STEERING_DIFF:
                    error = abs(steering_diff) - MAX_STEERING_DIFF
                    jerk_penalty += JERK_REWARD_WEIGHT * (error ** 2)
                else:
                    jerk_penalty += 0
        return jerk_penalty

    def postprocessing_step(self, action, observation, reward, done, info):
        """
        Update the reward (add jerk_penalty if needed), the command history
        and stack new observation (when using frame-stacking).
        :param action: ([float])
        :param observation: (np.ndarray)
        :param reward: (float)
        :param done: (bool)
        :param info: (dict)
        :return: (np.ndarray, float, bool, dict)
        """
        # Update command history
        if self.n_command_history > 0:
            self.command_history = np.roll(
                self.command_history, shift=-self.n_commands, axis=-1
            )
            self.command_history[..., -self.n_commands :] = action
            observation = np.concatenate((observation, self.command_history), axis=-1)

        jerk_penalty = 0  # self.jerk_penalty()
        # Cancel reward if the continuity constrain is violated
        if jerk_penalty > 0 and reward > 0:
            reward = 0
        reward -= jerk_penalty

        if self.n_stack > 1:
            self.stacked_obs = np.roll(
                self.stacked_obs, shift=-observation.shape[-1], axis=-1
            )
            if done:
                self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1] :] = observation
            return self.stacked_obs, reward, done, info

        return observation, reward, done, info

    