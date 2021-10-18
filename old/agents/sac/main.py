import gym
import gym
import time
from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# local 
from tools.all_tools import PATHS
from tools.reality_maker import MinimalAgent

import agents.sac.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


def sac(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    lr=1e-3,
    alpha=0.2,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_episode_length=1000,
    logger_kwargs=dict(),
    save_freq=1,
):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_episode_length (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    the_actor_critic = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    actor_critic_target = deepcopy(the_actor_critic)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for parameter in actor_critic_target.parameters():
        parameter.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_network_parameters = itertools.chain(the_actor_critic.q1.parameters(), the_actor_critic.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [the_actor_critic.pi, the_actor_critic.q1, the_actor_critic.q2])

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q1 = the_actor_critic.q1(o, a)
        q2 = the_actor_critic.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = the_actor_critic.pi(o2)

            # Target Q-values
            q1_pi_targ = actor_critic_target.q1(o2, a2)
            q2_pi_targ = actor_critic_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data["obs"]
        pi, logp_pi = the_actor_critic.pi(o)
        q1_pi = the_actor_critic.q1(o, pi)
        q2_pi = the_actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(the_actor_critic.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_network_parameters, lr=lr)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for parameter in q_network_parameters:
            parameter.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for parameter in q_network_parameters:
            parameter.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for parameter_from_actor_critic, parameter_from_target_actor_critic in zip(the_actor_critic.parameters(), actor_critic_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                parameter_from_target_actor_critic.data.mul_(polyak)
                parameter_from_target_actor_critic.data.add_((1 - polyak) * parameter_from_actor_critic.data)

    def get_action(o, deterministic=False):
        return the_actor_critic.act(torch.as_tensor(o, dtype=torch.float32), deterministic)
    
    timestep = None
    timeline = []
    observation, action, reward = None, None, None
    def when_episode_starts():
        nonlocal timestep, episode_return, observation, action, reward
        timestep = -1
        episode_return = 0
        observation, action, reward = None, None, None
    
    def when_timestep_happens(body):
        nonlocal timestep, episode_return, observation, action, reward
        timestep += 1
        
        # get the reward for the previous action
        prev_observation = observation
        prev_action = action
        reward = body.get_reward()
        observation = body.get_observation()
        if prev_action != None:
            episode_return += reward
            # save triplets
            timeline.append((prev_observation, prev_action, reward))
            # its never the end of the world (if it is, then this function wouldnt be called)
            replay_buffer.store(prev_observation, prev_action, prev_reward, observation, False)
            
            # Update handling
            if timestep >= update_after and timestep % update_every == 0:
                for _ in range(update_every):
                    update(data=replay_buffer.sample_batch(batch_size))
        # 
        # 
        # pick action
        # 
        # 
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if timestep > start_steps:
            action = get_action(observation)
        else:
            action = body.action_space.sample()
            
    def when_episode_ends(episode_index):
        nonlocal timestep, episode_return, observation, action, reward
        timestep += 1
        
        # get the reward for the previous action
        prev_observation = observation
        prev_action      = action
        reward           = body.get_reward()
        observation      = body.get_observation()
        if prev_action != None:
            episode_return += reward
            # save triplets
            timeline.append((prev_observation, prev_action, reward))
            # always the end of the world (thats the only time this function is called)
            replay_buffer.store(prev_observation, prev_action, prev_reward, observation, True)
            
            # Update handling
            if timestep >= update_after and timestep % update_every == 0:
                for _ in range(update_every):
                    update(data=replay_buffer.sample_batch(batch_size))    


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env", type=str, default="HalfCheetah-v2")
#     parser.add_argument("--hid", type=int, default=256)
#     parser.add_argument("--l", type=int, default=2)
#     parser.add_argument("--gamma", type=float, default=0.99)
#     parser.add_argument("--seed", "-s", type=int, default=0)
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--exp_name", type=str, default="sac")
#     args = parser.parse_args()

#     from spinup.utils.run_utils import setup_logger_kwargs

#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#     torch.set_num_threads(torch.get_num_threads())

#     sac(
#         lambda: gym.make(args.env),
#         actor_critic=core.MLPActorCritic,
#         ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
#         gamma=args.gamma,
#         seed=args.seed,
#         epochs=args.epochs,
#         logger_kwargs=logger_kwargs,
#     )



class Agent(MinimalAgent):
    def __init__(self, body_type, **config):
        super(Agent, self).__init__(body_type)
        # save config for later
        self.config = config
    
    def when_body_is_ready(self):
        # wrapper env
        agent = self
        class DummyEnv(gym.Env):
            @property
            def action_space(self): return agent.body.action_space
            @property
            def observation_space(self): return agent.body.observation_space
            
            def close(self): pass
            def step(self, action=None): return None, None, None, None
            def reset(self): return None
        
        self.spinup_sac = sac(
            env_fn=lambda: DummyEnv(),
            **self.config,
        )
        
    def when_campaign_starts(self):
        pass
        
    def when_episode_starts(self, episode_index):
        # FIXME: init stuff
        pass
    
    def when_timestep_happens(self):
        # FIXME: get new action
        self.spinup_sac
        
    def when_episode_ends(self, episode_index):
        # FIXME: update the state, and update the 
        pass
    
    def when_campaign_ends(self):
        pass