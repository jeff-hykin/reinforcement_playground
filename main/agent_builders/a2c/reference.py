from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
import torch

from tools.basics import product, flatten

# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax()
        )
    
    def forward(self, X):
        return self.model(X)
    

# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, X):
        return self.model(X)


class Agent():
    def __init__(self, ):
        self.observation_size = env.observation_space.shape[0]
        self.number_of_actions = env.action_space.n
        self.actor = Actor(self.observation_size, self.number_of_actions)
        self.critic = Critic(self.observation_size)
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.discount_factor = 0.99
        self.prev_action = None
        self.prev_action_choice_distribution = None
    
    def make_decision(self, observation):
        probs = self.actor(torch.from_numpy(observation).float())
        self.prev_action_choice_distribution = torch.distributions.Categorical(probs=probs)
        self.prev_action = self.prev_action_choice_distribution.sample()
        return self.prev_action.detach().data.numpy()
    
    def approximate_value_of(self, observation):
        return self.critic(torch.from_numpy(observation).float())
    
    def compute_advantage(self, *, reward, observation, next_observation, episode_is_over):
        return reward + \
            self.approximate_value_of(next_observation)*self.discount_factor*(1-int(episode_is_over))\
            - self.approximate_value_of(observation)
    
    def update_weights(self, advantage):
        critic_loss = advantage.pow(2).mean()
        self.adam_critic.zero_grad()
        critic_loss.backward()
        self.adam_critic.step()

        actor_loss = -self.prev_action_choice_distribution.log_prob(self.prev_action)*advantage.detach()
        self.adam_actor.zero_grad()
        actor_loss.backward()
        self.adam_actor.step()
    
    # @ConnectBody.when_episode_starts
    def when_episode_starts(self, episode_index):
        self.accumulated_reward = 0
        self.timestep = 0
        # 
        # logging
        # 
        self.record_keeper.pending_record["updated_weights"] = False
        print("{"+f' "episode": {str(episode_index).ljust(5)},', end="")
    
    # @ConnectBody.when_timestep_happens
    def when_timestep_happens(self, timestep_index):
        self.timestep = timestep_index
        # 
        # save the reward for the previous action
        # 
        reward = self.body.get_reward()
        self.accumulated_reward += reward
        self.record_keeper_by_update.pending_record["reward"] += reward
        self.brain.buffer.rewards.append(reward)
        self.brain.buffer.is_terminals.append(False)
        
        # 
        # occasionally update the network
        # 
        if self.countdown_till_update():
            print('"update":true,', end="")
            # save the existing record
            update_index = self.record_keeper_by_update.pending_record["update_index"]
            self.record_keeper_by_update.commit_record()
            # setup for the next record
            self.record_keeper_by_update.pending_record["update_index"] = update_index + 1
            self.record_keeper_by_update.pending_record["reward"] = 0
            # perform the update
            self.brain.update()
        
        # 
        # update action standard deviation
        # 
        # if continuous action space; then decay action std of ouput action distribution
        if self.brain.is_continuous_action_space:
            if self.countdown_till_decay():
                self.brain.decay_action_std(self.action_standard_deviation_decay_rate, self.minimum_action_standard_deviation)
        
        # 
        # take action!
        # 
        observation = self.body.get_observation()
        # brain decides action (reshape because the agent wants a batch)
        action_choice = self.brain.select_action(observation.reshape((1,*observation.shape)))
        # actually perform the action in the world
        self.body.perform_action(action_choice)
        
    # @ConnectBody.when_episode_ends
    def when_episode_ends(self, episode_index):
        # 
        # save last reward
        # 
        reward = self.body.get_reward()
        self.accumulated_reward += reward
        self.brain.buffer.rewards.append(reward)
        self.brain.buffer.is_terminals.append(True)
        
        # 
        # log
        #
        self.record_keeper.pending_record["accumulated_reward"] = self.accumulated_reward
        self.record_keeper.pending_record["episode_index"] = episode_index
        self.record_keeper.commit_record()
        formatted_reward   = f"{self.accumulated_reward:,.2f}".rjust(8)
        formatted_timestep = f"{self.timestep}".rjust(6)
        print(f' "rewardSum": {formatted_reward}, "numberOfTimesteps": {formatted_timestep}'+" },", flush=True)
        
        #
        # occasionally save the model
        #
        if self.countdown_till_checkpoint():
            print("saving model in " + self.save_folder)
            self.save_network()
            print("model saved")
    

env = gym.make("CartPole-v1")


# config
mr_bond = Agent()
episode_rewards = []

for i in range(500):
    done = False
    total_reward = 0
    observation = env.reset()


    while not done:
        next_observation, reward, done, info = env.step(mr_bond.make_decision(observation))
        advantage = mr_bond.compute_advantage(reward=reward,observation=observation, next_observation=next_observation, episode_is_over=done)
        
        total_reward += reward
        observation = next_observation

        mr_bond.update_weights(advantage)
            
    episode_rewards.append(total_reward)


ss.DisplayCard("quickScatter", episode_rewards)