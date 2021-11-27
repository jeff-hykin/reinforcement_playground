import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np
import gym

# local
from tools.pytorch_tools import device

from agent_builders.ppo.rollout_buffer import RolloutBuffer
from agent_builders.ppo.actor_critic import ActorCritic

class PpoBrain:
    def __init__(
        self,
        state_dim,
        action_dim,
        is_continuous_action_space,
        actor_learning_rate=0.0003,
        critic_learning_rate=0.001,
        discount_factor=0.99,
        number_of_epochs_to_optimize=40,
        loss_clamp_boundary=0.2,
        action_std_init=0.6,
    ):
        # Copy in values
        self.is_continuous_action_space  = is_continuous_action_space
        self.discount_factor              = discount_factor
        self.loss_clamp_boundary          = loss_clamp_boundary
        self.number_of_epochs_to_optimize = number_of_epochs_to_optimize

        # Setup Networks and tools
        self.buffer     = RolloutBuffer()
        self.policy     = ActorCritic(state_dim, action_dim, is_continuous_action_space, action_std_init).to(device)
        self.old_policy = ActorCritic(state_dim, action_dim, is_continuous_action_space, action_std_init).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': actor_learning_rate},
            {'params': self.policy.critic.parameters(), 'lr': critic_learning_rate}
        ])
        
        # Special logic
        if is_continuous_action_space:
            self.action_std = action_std_init

    def set_action_std(self, new_action_std):
        if self.is_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.old_policy.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PpoBrain::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if not self.is_continuous_action_space:
            print("WARNING : Calling PpoBrain::decay_action_std() on discrete action space policy")
        else:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.old_policy.act(state)
        # update buffer
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probabilities.append(action_logprob)
        # return action as value
        return action.item() if not self.is_continuous_action_space else action.detach().cpu().numpy().flatten()

    def update(self):
        self.buffer.equalize()
        
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.discount_factor * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_log_probabilities = torch.squeeze(torch.stack(self.buffer.log_probabilities, dim=0)).detach().to(device)
        
        # Optimize policy for K epochs
        for _ in range(self.number_of_epochs_to_optimize):
            # Evaluating old actions and values
            log_probabilities, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probabilities - old_log_probabilities.detach())
            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.loss_clamp_boundary, 1+self.loss_clamp_boundary) * advantages
            # final loss of clipped objective PpoBrain
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            # gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy, then reset the buffer
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.old_policy.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.old_policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
