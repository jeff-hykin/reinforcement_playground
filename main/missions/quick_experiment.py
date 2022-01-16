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

# 22 min: Episode 1000 score = 9.0, average score = 5.572

from world_builders.atari.medium_preprocessor import create_env
from agent_builders.medium_dqn.main import Agent

def run(training_mode, pretrained, path="./models.ignore/medium_dqn", num_episodes=1000, exploration_max=1):
    path += "/"
    
    env = create_env(
        gym.make('Breakout-v0')
    )
    
    agent = Agent(
        state_space=env.observation_space.shape,
        action_space=env.action_space.n,
        max_memory_size=30000,
        batch_size=32,
        gamma=0.90,
        lr=0.00025,
        dropout=0.2,
        exploration_max=1.0,
        exploration_min=0.02,
        exploration_decay=0.99,
        pretrained=pretrained,
        path=path,
    )
    
    total_rewards = []
    if training_mode and pretrained:
        with open(path+"total_rewards.pkl", 'rb') as f:
            total_rewards = pickle.load(f)
    
    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        while True:
            action = agent.act(state)
            steps += 1
            
            state_next, reward, episode_is_over, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            
            episode_is_over = torch.tensor([int(episode_is_over)]).unsqueeze(0)
            
            if training_mode:
                agent.remember(state, action, reward, state_next, episode_is_over)
                agent.experience_replay()
            
            state = state_next
            if episode_is_over:
                break
        
        total_rewards.append(total_reward)
        
        if ep_num != 0 and ep_num % 100 == 0:
            print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1], np.mean(total_rewards)))
        num_episodes += 1  

    print("Episode {} score = {}, average score = {}".format(ep_num + 1, total_rewards[-1], np.mean(total_rewards)))
    
    # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
    if training_mode:
        with open(path+"ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open(path+"num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open(path+"total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)


        torch.save(agent.dqn.state_dict(), path+"DQN.pt")  
        torch.save(agent.STATE_MEM,  path+"STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, path+"ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, path+"REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, path+"STATE2_MEM.pt")
        torch.save(agent.DONE_MEM,   path+"DONE_MEM.pt")
    
    env.close()


run(training_mode=True, pretrained=False, num_episodes=1000, exploration_max=1)