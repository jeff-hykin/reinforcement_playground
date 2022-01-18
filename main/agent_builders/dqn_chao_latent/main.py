import torch
import torch.nn as nn
import random
from tqdm import tqdm
import pickle 
import gym
import numpy as np
import collections 
import cv2
import time

from tools.agent_skeleton import Skeleton
from tools.file_system_tools import FileSystem
from tools.basics import product
from tools.pytorch_tools import opencv_image_to_torch_image, to_tensor, init, forward, Sequential, tensor_to_image
from tools.schedulers import BasicLearningRateScheduler

class Network(nn.Module):
    """
    Shallow Neural Net
    """
    @init.hardware
    def __init__(self, input_shape, n_actions):
        super(Network, self).__init__()
        self.input_shape = input_shape
        self.fc = Sequential(
            nn.Linear(product(self.input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    @forward.to_tensor
    @forward.to_device
    def forward(self, x):
        return self.fc(x).view(x.size()[0], -1)

class Agent(Skeleton):
    @init.hardware
    def __init__(self, observation_space, action_space, max_memory_size, batch_size, gamma, lr, dropout, exploration_max, exploration_min, exploration_decay, pretrained, path, training_mode):
        self.training_mode = training_mode
        self.path = path
        FileSystem.ensure_is_folder(FileSystem.dirname(self.path))

        # Define DQN Layers
        self.state_space = observation_space.shape
        self.action_space = action_space.n
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # DQN network  
        self.dqn = Network(self.state_space, self.action_space).to(self.device)

        if self.pretrained:
            self.dqn.load_state_dict(torch.load(self.path+"DQN.pt", map_location=torch.device(self.device)))
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load(self.path+"STATE_MEM.pt")
            self.ACTION_MEM = torch.load(self.path+"ACTION_MEM.pt")
            self.REWARD_MEM = torch.load(self.path+"REWARD_MEM.pt")
            self.STATE2_MEM = torch.load(self.path+"STATE2_MEM.pt")
            self.DONE_MEM = torch.load(self.path+"DONE_MEM.pt")
            with open(self.path+"ending_position.pkl", 'rb') as f:
                self.ending_position = pickle.load(f)
            with open(self.path+"num_in_queue.pkl", 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0
        
        self.memory_sample_size = batch_size

    def remember(self, state, action, reward, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)
    
    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]      
        return STATE, ACTION, REWARD, STATE2, DONE
    
    def act(self, state):
        """Epsilon-greedy action"""
        if random.random() < self.exploration_rate:  
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
    
    def experience_replay(self):
        if self.memory_sample_size > self.num_in_queue:
            return
    
        # Sample a batch of experiences
        STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)
        
        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
        target = REWARD + torch.mul((self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.dqn(STATE).gather(1, ACTION.long())
        
        loss = self.l1(current, target)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error

        self.exploration_rate *= self.exploration_decay
        
        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

    def save(self):
        if self.training_mode:
            with open(self.path+"ending_position.pkl", "wb") as f:
                pickle.dump(self.ending_position, f)
            with open(self.path+"num_in_queue.pkl", "wb") as f:
                pickle.dump(self.num_in_queue, f)
            with open(self.path+"total_rewards.pkl", "wb") as f:
                pickle.dump(self.total_rewards, f)

            torch.save(self.dqn.state_dict(), self.path+"DQN.pt")  
            torch.save(self.STATE_MEM,  self.path+"STATE_MEM.pt")
            torch.save(self.ACTION_MEM, self.path+"ACTION_MEM.pt")
            torch.save(self.REWARD_MEM, self.path+"REWARD_MEM.pt")
            torch.save(self.STATE2_MEM, self.path+"STATE2_MEM.pt")
            torch.save(self.DONE_MEM,   self.path+"DONE_MEM.pt")
    
    
    # 
    # hooks (special names)
    # 
    def when_mission_starts(self, mission_index=0):
        pass
    
    def when_episode_starts(self, episode_index):
        self.total_reward = 0
    
    def when_timestep_starts(self, timestep_index):
        self.action = self.act(self.observation)
        self.prev_observation = self.observation
    
    def when_timestep_ends(self, timestep_index):
        if self.training_mode:
            self.remember(self.prev_observation, self.action, self.reward, self.observation, self.episode_is_over)
            self.experience_replay()
        
        self.total_reward += self.reward
    
    def when_episode_ends(self, episode_index):
        self.total_rewards.append(self.total_reward)
        
        if episode_index != 0 and episode_index % 100 == 0:
            print("Episode {} score = {}, average score = {}".format(episode_index + 1, self.total_rewards[-1], np.mean(self.total_rewards)))
    
    def when_mission_ends(self, mission_index=0):
        pass
    
    def update_weights(self):
        pass
        
    