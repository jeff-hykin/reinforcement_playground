from typing import Any, Dict, Optional, Type, Union

import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance

from time import time
from super_map import LazyDict

from tools.debug import debug, ic
from tools.agent_skeleton import Skeleton


class Agent(OnPolicyAlgorithm, Skeleton):
    """
        Advantage Actor Critic (A2C)

        Paper: https://arxiv.org/abs/1602.01783
        Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
        and Stable Baselines (https://github.com/hill-a/stable-baselines)

        Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

        :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        :param env: The environment to learn from (if registered in Gym, can be str)
        :param learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        :param n_steps: The number of steps to run for each environment per update
            (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
        :param gamma: Discount factor
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            Equivalent to classic advantage when set to 1.
        :param ent_coef: Entropy coefficient for the loss calculation
        :param vf_coef: Value function coefficient for the loss calculation
        :param max_grad_norm: The maximum value for the gradient clipping
        :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
            of RMSProp update
        :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
        :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
            instead of action noise exploration (default: False)
        :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
            Default: -1 (only sample at the beginning of the rollout)
        :param normalize_advantage: Whether to normalize or not the advantage
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
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super(Agent, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()
        
        self.logging = Agent.Logger(agent=self)

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # values are value estimates from the network
            # log_prob is a vector of log probabilities of chosing each action
            # entropy is the vec of entropy values from each of the action distributions
            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()
            debug.rollout_data = rollout_data # FIXME: Debugging 
            debug.actions      = actions # FIXME     : Debugging 
            debug.values       = values # FIXME      : Debugging 
            debug.log_prob     = log_prob # FIXME    : Debugging 
            debug.entropy      = entropy # FIXME     : Debugging 

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "A2C",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "A2C":

        return super(A2C, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
    
    
    # 
    # Hooks (Special Names)
    # 
    def when_mission_starts(self):
        self.logging.when_mission_starts()
        
    def when_episode_starts(self, episode_index):
        self.logging.when_episode_starts(episode_index)
        
    def when_timestep_starts(self, timestep_index):
        self.logging.when_timestep_starts(timestep_index)
        self.action, _ = self.predict(self.observation)
    
    def when_timestep_ends(self, timestep_index):
        self.logging.when_timestep_ends(timestep_index)
    
    def when_episode_ends(self, episode_index):
        self.logging.when_episode_ends(episode_index)
    
    def when_mission_ends(self):
        self.logging.when_mission_ends()
    
    # 
    # tools
    # 
    class Logger:
        # depends on:
        #     self.agent.reward
        #     self.agent.loss
        def __init__(self, agent, **config):
            self.agent = agent
            
            self.should_display   = config.get("should_display"  , False)
            self.live_updates     = config.get("live_updates"    , False)
            self.smoothing_amount = config.get("smoothing_amount", 5    )
            self.episode_rewards = []
            self.episode_losses  = []
            self.episode_reward_card = None
            self.episode_loss_card = None
            self.number_of_updates = 0
            
            # init class attributes if doesn't already have them
            self.static = Agent.Logger.static = LazyDict(
                agent_number_count=0,
                total_number_of_episodes=0,
                total_number_of_timesteps=0,
                start_time=time(),
            ) if not hasattr(Agent.Logger, "static") else Agent.Logger.static
            
            # agent number count
            self.static.agent_number_count += 1
            self.agent_number = self.static.agent_number_count
            
        def when_mission_starts(self):
            self.episode_rewards.clear()
            self.episode_losses.clear()
            if self.live_updates:
                self.episode_loss_card = ss.DisplayCard("quickLine",[])
                ss.DisplayCard("quickMarkdown", f"#### Live {self.agent_number}: ⬆️ Loss, ➡️ Per Episode")
                self.episode_reward_card = ss.DisplayCard("quickLine",[])
                ss.DisplayCard("quickMarkdown", f"#### Live {self.agent_number}: ⬆️ Rewards, ➡️ Per Episode")
            
        def when_episode_starts(self, episode_index):
            self.accumulated_reward = 0
            self.accumulated_loss   = 0
            self.static.total_number_of_episodes += 1
        
        def when_timestep_starts(self, timestep_index):
            self.static.total_number_of_timesteps += 1
            
        def when_timestep_ends(self, timestep_index):
            self.accumulated_reward += self.agent.reward
        
        def when_episode_ends(self, episode_index):
            # logging
            self.episode_rewards.append(self.accumulated_reward)
            self.episode_losses.append(self.accumulated_loss)
            if self.live_updates:
                self.episode_reward_card.send     ([episode_index, self.accumulated_reward      ])
                self.episode_loss_card.send ([episode_index, self.accumulated_loss  ])
                print('episode_index = ', episode_index)
                print(f'    total_number_of_timesteps :{self.static.total_number_of_timesteps}',)
                print(f'    number_of_updates         :{self.number_of_updates}',)
                print(f'    average_episode_time      :{(time()-self.static.start_time)/self.static.total_number_of_episodes}',)
                print(f'    accumulated_reward        :{self.accumulated_reward      }',)
                print(f'    accumulated_loss          :{self.accumulated_loss  }',)
        
        def when_mission_ends(self,):
            if self.should_display:
                # graph reward results
                ss.DisplayCard("quickLine", stat_tools.rolling_average(self.episode_losses, self.smoothing_amount))
                ss.DisplayCard("quickMarkdown", f"#### {self.agent_number}: Losses Per Episode")
                ss.DisplayCard("quickLine", stat_tools.rolling_average(self.episode_rewards, self.smoothing_amount))
                ss.DisplayCard("quickMarkdown", f"#### {self.agent_number}: Rewards Per Episode")
        
        def when_weight_update_starts(self):
            self.number_of_updates += 1

        def when_weight_update_ends(self):
            self.accumulated_loss += self.agent.loss.item()
    
    