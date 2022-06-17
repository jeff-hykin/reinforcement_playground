from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from time import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Iterable
import io
import pathlib
import time
import warnings
import warnings

import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import utils
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, EvalCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import check_for_correct_spaces, get_device, get_schedule_fn, get_system_info, set_random_seed, zip_strict
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation, polyak_update
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import DummyVecEnv,VecEnv,VecNormalize,VecTransposeImage,is_vecenv_wrapped,unwrap_vec_normalize
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy

from super_map import LazyDict

from tools.debug import debug, ic
from tools.agent_skeleton import Skeleton

def correctness_check(self, policy, support_multi_env, supported_action_spaces):
    if supported_action_spaces is not None:
        assert isinstance(self.action_space, supported_action_spaces), (
            f"The algorithm only supports {supported_action_spaces} as action spaces "
            f"but {self.action_space} was provided"
        )

    if not support_multi_env and self.n_envs > 1:
        raise ValueError(
            "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
        )

    # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
    if policy in ["MlpPolicy", "CnnPolicy"] and isinstance(self.observation_space, gym.spaces.Dict):
        raise ValueError(f"You must use `MultiInputPolicy` when working with dict observation space, not {policy}")

    if isinstance(self.action_space, gym.spaces.Box):
        assert np.all(
            np.isfinite(np.array([self.action_space.low, self.action_space.high]))
        ), "Continuous action space must have a finite lower and upper bound"


def generate_replay_buffer(env, observation_space, action_space, n_envs, buffer_class, buffer, buffer_size, device, buffer_kwargs, optimize_memory_usage):
    # Use DictReplayBuffer if needed
    if buffer_class is None:
        buffer_class = DictReplayBuffer     if isinstance(observation_space, gym.spaces.Dict)     else ReplayBuffer
    elif buffer_class == HerReplayBuffer:
        assert env is not None, "You must pass an environment when using `HerReplayBuffer`"
        buffer = HerReplayBuffer(
            env,
            buffer_size,
            device=device,
            # If using offline sampling, we need a classic replay buffer too
            buffer=(None if buffer_kwargs.get("online_sampling", True) else DictReplayBuffer(
                buffer_size,
                observation_space,
                action_space,
                device=device,
                optimize_memory_usage=optimize_memory_usage,
            )),
            **buffer_kwargs,
        )
    if buffer is None:
        buffer = buffer_class(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            **buffer_kwargs,
        )
    return buffer

def ensure_train_frequency_object(train_freq):
    """
    Convert `train_freq` parameter (int or tuple)
    to a TrainFreq object.
    """
    if not isinstance(train_freq, TrainFreq):

        # The value of the train frequency will be checked later
        if not isinstance(train_freq, tuple):
            train_freq = (train_freq, "step")

        try:
            train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
        except ValueError:
            raise ValueError(f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

        if not isinstance(train_freq[0], int):
            raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

        train_freq = TrainFreq(*train_freq)
    
    return train_freq

def number_of_envs_warning(n_envs, target_update_interval):
    # Account for multiple environments
    # each call to step() corresponds to n_envs transitions
    if n_envs > 1 and n_envs > target_update_interval:
        warnings.warn(
            "The number of environments used is greater than the target network "
            f"update interval ({n_envs} > {target_update_interval}), "
            "therefore the target network will be updated after each call to env.step() "
            f"which corresponds to {n_envs} steps."
        )

def init_random_seed(seed=None, envs=[], device=None, action_space=None):
    """
    Set the seed of the pseudo-random generators
    (python, numpy, pytorch, gym, action_space)

    :param seed:
    """
    from stable_baselines3.common.utils import set_random_seed
    
    if seed is None:
        return
    set_random_seed(seed, using_cuda=device.type == th.device("cuda").type)
    action_space.seed(seed)
    for env in envs:
        # TODO: shouldnt all the action spaces be seeded?
        if env is not None:
           env.seed(seed)


def wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
    """ "
    Wrap environment with the appropriate wrappers if needed.
    For instance, to have a vectorized environment
    or to re-order the image channels.

    :param env:
    :param verbose:
    :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
    :return: The wrapped environment.
    """
    if env is None:
        return None
    
    if not isinstance(env, VecEnv):
        if not is_wrapped(env, Monitor) and monitor_wrapper:
            if verbose >= 1: print("Wrapping the env with a `Monitor` wrapper")
            env = Monitor(env)
        
        if verbose >= 1: print("Wrapping the env in a DummyVecEnv.")
        env = DummyVecEnv([lambda: env])

    # Make sure that dict-spaces are not nested (not supported)
    check_for_nested_spaces(env.observation_space)

    if not is_vecenv_wrapped(env, VecTransposeImage):
        wrap_with_vectranspose = False
        if isinstance(env.observation_space, gym.spaces.Dict):
            # If even one of the keys is a image-space in need of transpose, apply transpose
            # If the image spaces are not consistent (for instance one is channel first,
            # the other channel last), VecTransposeImage will throw an error
            for space in env.observation_space.spaces.values():
                wrap_with_vectranspose = wrap_with_vectranspose or (
                    is_image_space(space) and not is_image_space_channels_first(space)
                )
        else:
            wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                env.observation_space
            )

        if wrap_with_vectranspose:
            if verbose >= 1:
                print("Wrapping the env in a VecTransposeImage.")
            env = VecTransposeImage(env)

    return env
        
class DQN:
    """
        Deep Q-Network (DQN)

        Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
        Default hyperparameters are taken from the nature paper,
        except for the optimizer and learning rate that were taken from Stable Baselines defaults.

        :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        :param env: The environment to learn from (if registered in Gym, can be str)
        :param learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        :param buffer_size: size of the replay buffer
        :param learning_starts: how many steps of the model to collect transitions for before learning starts
        :param batch_size: Minibatch size for each gradient update
        :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
        :param gamma: the discount factor
        :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
            like ``(5, "step")`` or ``(2, "episode")``.
        :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
            Set to ``-1`` means to do as many gradient steps as steps done in the environment
            during the rollout.
        :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
            If ``None``, it will be automatically selected.
        :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
        :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
            at a cost of more complexity.
            See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        :param target_update_interval: update the target network every ``target_update_interval``
            environment steps.
        :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
        :param exploration_initial_eps: initial value of random action probability
        :param exploration_final_eps: final value of random action probability
        :param max_grad_norm: The maximum value for the gradient clipping
        :param tensorboard_log: the log location for tensorboard (if None, no logging)
        :param create_eval_env: Whether to create a second environment that will be
            used for evaluating the agent periodically. (Only available when passing string for the environment)
        :param policy_kwargs: additional arguments to be passed to the policy on creation
        :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
        :param seed: Seed for the pseudo random generators
        :param device: Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible.
        :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """
        # """
        #     The base for Off-Policy algorithms (ex: SAC/TD3)
        #
        #     :param policy: Policy object
        #     :param env: The environment to learn from
        #                 (if registered in Gym, can be str. Can be None for loading trained models)
        #     :param learning_rate: learning rate for the optimizer,
        #         it can be a function of the current progress remaining (from 1 to 0)
        #     :param buffer_size: size of the replay buffer
        #     :param learning_starts: how many steps of the model to collect transitions for before learning starts
        #     :param batch_size: Minibatch size for each gradient update
        #     :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
        #     :param gamma: the discount factor
        #     :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        #         like ``(5, "step")`` or ``(2, "episode")``.
        #     :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        #         Set to ``-1`` means to do as many gradient steps as steps done in the environment
        #         during the rollout.
        #     :param action_noise: the action noise type (None by default), this can help
        #         for hard exploration problem. Cf common.noise for the different action noise type.
        #     :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        #         If ``None``, it will be automatically selected.
        #     :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
        #     :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        #         at a cost of more complexity.
        #         See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        #     :param policy_kwargs: Additional arguments to be passed to the policy on creation
        #     :param tensorboard_log: the log location for tensorboard (if None, no logging)
        #     :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
        #     :param device: Device on which the code should run.
        #         By default, it will try to use a Cuda compatible device and fallback to cpu
        #         if it is not possible.
        #     :param support_multi_env: Whether the algorithm supports training
        #         with multiple environments (as in A2C)
        #     :param create_eval_env: Whether to create a second environment that will be
        #         used for evaluating the agent periodically. (Only available when passing string for the environment)
        #     :param monitor_wrapper: When creating an environment, whether to wrap it
        #         or not in a Monitor wrapper.
        #     :param seed: Seed for the pseudo random generators
        #     :param sde_support: Whether the model support gSDE or not
        #     :param supported_action_spaces: The action spaces supported by the algorithm.
        # """
        
            # """
            # The base of RL algorithms
            #
            # :param policy: Policy object
            # :param env: The environment to learn from
            #             (if registered in Gym, can be str. Can be None for loading trained models)
            # :param learning_rate: learning rate for the optimizer,
            #     it can be a function of the current progress remaining (from 1 to 0)
            # :param policy_kwargs: Additional arguments to be passed to the policy on creation
            # :param tensorboard_log: the log location for tensorboard (if None, no logging)
            # :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
            # :param device: Device on which the code should run.
            #     By default, it will try to use a Cuda compatible device and fallback to cpu
            #     if it is not possible.
            # :param support_multi_env: Whether the algorithm supports training
            #     with multiple environments (as in A2C)
            # :param create_eval_env: Whether to create a second environment that will be
            #     used for evaluating the agent periodically. (Only available when passing string for the environment)
            # :param monitor_wrapper: When creating an environment, whether to wrap it
            #     or not in a Monitor wrapper.
            # :param seed: Seed for the pseudo random generators
            # :param supported_action_spaces: The action spaces supported by the algorithm.
            # """
    
    # Policy aliases (see _get_policy_from_name())
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    
    def __init__(
        self, 
        policy                 : Union[str, Type[DQNPolicy]], 
        env                    : Union[GymEnv, str], 
        learning_rate          : Union[float, Schedule]      = 1e-4     , 
        buffer_size            : int                         = 1_000_000, # 1e6
        learning_starts        : int                         = 50000    , 
        batch_size             : int                         = 32       , 
        tau                    : float                       = 1.0      , 
        gamma                  : float                       = 0.99     , 
        train_freq             : Union[int, Tuple[int, str]] = 4        , 
        gradient_steps         : int                         = 1        , 
        replay_buffer_class    : Optional[ReplayBuffer]      = None     , 
        replay_buffer_kwargs   : Optional[Dict[str, Any]]    = None     , 
        optimize_memory_usage  : bool                        = False    , 
        target_update_interval : int                         = 10000    , 
        exploration_fraction   : float                       = 0.1      , 
        exploration_initial_eps: float                       = 1.0      , 
        exploration_final_eps  : float                       = 0.05     , 
        max_grad_norm          : float                       = 10       , 
        tensorboard_log        : Optional[str]               = None     , 
        create_eval_env        : bool                        = False    , 
        policy_kwargs          : Optional[Dict[str, Any]]    = None     , 
        verbose                : int                         = 0        , 
        seed                   : Optional[int]               = None     , 
        device                 : Union[th.device , str]      = "auto"   , 
        _init_setup_model      : bool                        = True     , 
    ):
        action_noise            = None  # No action noise
        sde_support             = False
        supported_action_spaces = (gym.spaces.Discrete,)
        support_multi_env       = True
        monitor_wrapper         = True
        
        if True: # OffPolicyAlgorithm
            if True: # BaseAlgorithm
                self.env                         = None  
                self.eval_env                    = None
                self.observation_space           = None  # Optional[gym.spaces.Space]
                self.action_space                = None  # Optional[gym.spaces.Space]
                self.n_envs                      = None
                self.action_noise                = None  # Optional[ActionNoise]
                self.start_time                  = None
                self.policy                      = None
                self.lr_schedule                 = None  # Optional[Schedule]
                self._last_obs                   = None  # Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
                self._last_episode_starts        = None  # Optional[np.ndarray]
                self._last_original_obs          = None  # When using VecNormalize                        : 
                self.ep_info_buffer              = None  # Buffers for logging, Optional[deque]
                self.ep_success_buffer           = None  # Buffers for logging, Optional[deque]
                self._logger                     = None  
                self._episode_num                = 0
                self._current_progress_remaining = 1     # Track the training progress remaining (from 1 to 0), this is used to update the learning rate
                self._n_updates                  = 0     # For logging (and TD3 delayed updates): int
                self.num_timesteps               = 0
                self._total_timesteps            = 0 # Used for updating schedules
                self._num_timesteps_at_start     = 0 # Used for computing fps , it is updated at each call of learn()
                self._custom_logger              = False
                self.verbose                     = verbose
                self.learning_rate               = learning_rate
                self.seed                        = seed
                self.tensorboard_log             = tensorboard_log
                self.policy_kwargs               = {} if policy_kwargs is None else policy_kwargs
                self.policy_class                = policy       if not isinstance(policy, str)    else     self.policy_aliases[policy]
                self.device                      = get_device(device); print(f"Using {self.device} device") if verbose > 0 else None
                self._vec_normalize_env          = unwrap_vec_normalize(env)

                # Create and wrap the env if needed
                if env is not None:
                    self.env               = gym.make(env)   if isinstance(env, str)                       else  env
                    self.eval_env          = self.env        if isinstance(env, str) and create_eval_env   else  None
                    self.env               = wrap_env(env, self.verbose, monitor_wrapper)
                    self.observation_space = self.env.observation_space
                    self.action_space      = self.env.action_space
                    self.n_envs            = self.env.num_envs
                    
                    correctness_check(self, policy, support_multi_env, supported_action_spaces)

            self._episode_storage      = None
            self.actor                 = None  # type: Optional[th.nn.Module]
            self.replay_buffer         = None  # type: Optional[ReplayBuffer]
            self.buffer_size           = buffer_size
            self.batch_size            = batch_size
            self.learning_starts       = learning_starts
            self.tau                   = tau
            self.gamma                 = gamma
            self.gradient_steps        = gradient_steps
            self.action_noise          = action_noise
            self.optimize_memory_usage = optimize_memory_usage
            self.train_freq            = train_freq # Save train freq parameter, will be converted later to TrainFreq object
            self.replay_buffer_class   = replay_buffer_class
            self.replay_buffer_kwargs  = {} if replay_buffer_kwargs is None else replay_buffer_kwargs

        self.exploration_schedule    = None # Linear schedule will be defined in `_setup_model()`
        self.q_net                   = None
        self.q_net_target            = None
        self._n_calls                = 0 # For updating the target network with multiple envs: 
        self.exploration_rate        = 0.0 # "epsilon" for the epsilon-greedy exploration
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps   = exploration_final_eps
        self.exploration_fraction    = exploration_fraction
        self.target_update_interval  = target_update_interval
        self.max_grad_norm           = max_grad_norm
        
        self.torch_save_params = ( ["policy", "policy.optimizer"], [] )
        self._get_torch_save_params = lambda : self.torch_save_params
        self.get_env                = lambda : self.env # for backwards compatibility
        self.get_vec_normalize_env  = lambda : self._vec_normalize_env
        self.get_parameters         = lambda : { name: recursive_getattr(self, name).state_dict() for name in self.torch_save_params[0] }

        if _init_setup_model: self._setup_model()


    @property
    def logger(self) -> Logger:
        return self._logger
    
    def set_logger(self, logger):
        self._logger = logger
        self._custom_logger = True
    
    @staticmethod
    def _wrap_env(*args, **kwargs):
        return wrap_env(*args, **kwargs)
    
    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""
        init_random_seed(
            seed=self.seed,
            envs=[ self.env, self.eval_env ],
            device=None,
            action_space=None,
        )

        self.train_freq             = ensure_train_frequency_object(self.train_freq) # Convert parameter to object
        self.lr_schedule            = get_schedule_fn(self.learning_rate) # ensure its a schedule object
        self.exploration_schedule   = get_linear_fn(self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction)
        self.replay_buffer          = generate_replay_buffer(self.env, self.observation_space, self.action_space, self.n_envs, self.replay_buffer_class, self.replay_buffer, self.buffer_size, self.device, self.replay_buffer_kwargs, self.optimize_memory_usage)
        self.policy                 = self.policy_class(self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs).to(self.device)
        self.q_net                  = self.policy.q_net
        self.q_net_target           = self.policy.q_net_target
        self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)    if self.n_envs > 1     else self.target_update_interval
        
        number_of_envs_warning(self.n_envs, self.target_update_interval)

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        # 
        # UPDATE WEIGHTS CHECK
        # 
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            # polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            params = self.q_net_target.parameters()
            target_params = self.q_net_target.parameters()
            with th.no_grad():
                for param, target_param in zip_strict(params, target_params): # zip does not raise an exception if length of parameters does not match.
                    target_param.data.mul_(1 - self.tau)
                    th.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)
        
        # 
        # exploration update
        # 
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        
        # 
        # logging
        # 
        self.logger.record("rollout/exploration_rate", self.exploration_rate)
    
    # 
    # update learning rate
    # 
    def _update_learning_rate(self, optimizers) -> None:
        """
            Update the optimizers learning rate using the current learning rate schedule
            and the current progress remaining (from 1 to 0).

            :param optimizers:
                An optimizer or a list of optimizers.
        """
        # 
        # logging
        # 
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))
        
        # make iterable
        optimizers = [optimizers]  if not isinstance(optimizers, list) else optimizers
        
        # update them
        for optimizer in optimizers:
            learning_rate = self.lr_schedule(self._current_progress_remaining)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(self.observation_space, gym.spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ):
        """
        Return a trained model.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param eval_env: Environment that will be used to evaluate the agent
        :param eval_freq: Evaluate the agent every ``eval_freq`` timesteps (this may vary a little)
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param eval_log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :return: the trained model
        """
        
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space
        - action_space

        :param env: The environment for learning a policy
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        """
        # if it is not a VecEnv, make it a VecEnv
        # and do other transformations (dict obs, image transpose) if needed
        env = wrap_env(env, self.verbose)
        # Check that the observation spaces match
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        # Update VecNormalize object
        # otherwise the wrong env may be used, see https://github.com/DLR-RM/stable-baselines3/issues/637
        self._vec_normalize_env = unwrap_vec_normalize(env)

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs = None

        self.n_envs = env.num_envs
        self.env = env

    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # We are only interested in former here.
        objects_needing_update = set(self.torch_save_params[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.")

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    @classmethod
    def load(
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> "BaseAlgorithm":
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path, device=device, custom_objects=custom_objects, print_system_info=print_system_info
        )

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            env = wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # put state_dicts back in place
        model.set_parameters(params, exact_match=True, device=device)

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        return model

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self.torch_save_params
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)

    def _init_callback(
        self,
        callback: MaybeCallback,
        eval_env: Optional[VecEnv] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations; if None, do not evaluate.
        :param n_eval_episodes: How many episodes to play per evaluation
        :param n_eval_episodes: Number of episodes to rollout during evaluation.
        :param log_path: Path to a folder where the evaluations will be saved
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Create eval callback in charge of the evaluation
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=log_path,
                log_path=log_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
            )
            callback = CallbackList([callback, eval_callback])

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46

        # Special case when using HerReplayBuffer,
        # the classic replay buffer is inside it when using offline sampling
        if isinstance(self.replay_buffer, HerReplayBuffer):
            replay_buffer = self.replay_buffer.replay_buffer
        else:
            replay_buffer = self.replay_buffer

        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and replay_buffer is not None
            and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True
        
        
        self.start_time = time.time()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            self._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        if eval_env is not None and self.seed is not None:
            eval_env.seed(self.seed)

        eval_env = wrap_env(eval_env or self.eval_env, self.verbose)

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)

        return total_timesteps, callback

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts:
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time_elapsed + 1e-8))
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # 
            # update info buffer
            # 
            if dones is None:
                dones = np.array([False] * len(infos))
            for idx, info in enumerate(infos):
                maybe_ep_info = info.get("episode")
                maybe_is_success = info.get("is_success")
                if maybe_ep_info is not None:
                    self.ep_info_buffer.extend([maybe_ep_info])
                if maybe_is_success is not None and dones[idx]:
                    self.ep_success_buffer.append(maybe_is_success)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            # (starts from 1 and ends to 0)
            self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    
class Agent(Skeleton):
    def __init__(self, 
        observation_space,
        action_space,
        parameters=None,
        weights=None,
        settings=None,
    ):
        self.parameters = LazyDict(
                learning_rate=0.5,
                discount_factor=0.9,
                epsilon=1.0,
                epsilon_decay=0.001,
            ).merge(parameters)
        self.weights = LazyDict(
                q_lookup=np.zeros((self.observation_space.n, self.action_space.n)),
            ).merge(weights)
        self.settings= LazyDict(
                training=True,
            ).merge(settings)


if __name__ == "__main__":
    import gym

    env = gym.make("CartPole-v1")

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()