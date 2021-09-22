import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random
from super_map import LazyDict
from tools.pytorch_tools import to_tensor

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            observation, action, reward, next_observation, done = transition
            s_lst.append(observation)
            a_lst.append([action])
            r_lst.append([reward])
            s_prime_lst.append(next_observation)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, hyperparameters):
        super(PolicyNet, self).__init__()
        self.hyperparameters = LazyDict(
            learning_rate         = 0.0005,
            alt_net_init          = 0.01,
            alt_net_learning_rate = 0.001,
            target_entropy        = -1.0, # for automated alpha update
        ).merge(hyperparameters or {})
        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.hyperparameters.learning_rate)

        self.log_alpha = torch.tensor(np.log(self.hyperparameters.alt_net_init))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.hyperparameters.alt_net_learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        observation, _, _, _, _ = mini_batch
        action, log_prob = self.forward(observation)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(observation,action), q2(observation,action)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.hyperparameters.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self, hyperparameters):
        super(QNet, self).__init__()
        self.hyperparameters = LazyDict(
            learning_rate = 0.001,
            tau           = 0.01,
        ).merge(hyperparameters or {})
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_cat = nn.Linear(128,32)
        self.fc_out = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.hyperparameters.learning_rate)

    def forward(self, x, action):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(action))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        observation, action, reward, next_observation, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(observation, action) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.hyperparameters.tau) + param.data * self.hyperparameters.tau)

def calc_target(policy_network, q1, q2, mini_batch, discount_rate):
    observation, action, reward, next_observation, done = mini_batch

    with torch.no_grad():
        next_action, log_prob= policy_network(next_observation)
        entropy = -policy_network.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(next_observation,next_action), q2(next_observation,next_action)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = reward + discount_rate * done * (min_q + entropy)

    return target


class Agent:
    def __init__(self, action_space=None, hyperparameters=None, **config):
        """
        arguments:
        """
        #
        # standard agent stuff
        # 
        self.config = config
        self.action_space = action_space
        self.wants_to_quit = False
        self.show = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
        
        # 
        # SAC stuff
        #
        self.print_interval = self.config.get("print_interval", 20)
        self.hyperparameters = LazyDict(
            # default values overridden by args
            policy_learning_rate          = 0.0005,
            critic_learning_rate          = 0.001,
            alt_net_init                  = 0.01,
            alt_net_learning_rate         = 0.001,  # for automated alpha update
            discount_rate                 = 0.98,
            batch_size                    = 32,
            buffer_limit                  = 50000,
            tau                           = 0.01, # for target network soft update
            target_entropy                = -1.0, # for automated alpha update
            minimum_traning_size          = 1000,
            minibatch_training_iterations = 20,
        ).merge(hyperparameters or {})
        
        self.memory = ReplayBuffer(buffer_limit=self.hyperparameters.buffer_limit)
        self.q1        = QNet(dict(learning_rate=self.hyperparameters.critic_learning_rate, tau=self.hyperparameters.tau))
        self.q2        = QNet(dict(learning_rate=self.hyperparameters.critic_learning_rate, tau=self.hyperparameters.tau))
        self.q1_target = QNet(dict(learning_rate=self.hyperparameters.critic_learning_rate, tau=self.hyperparameters.tau))
        self.q2_target = QNet(dict(learning_rate=self.hyperparameters.critic_learning_rate, tau=self.hyperparameters.tau))
        self.policy = PolicyNet(dict(
            learning_rate=self.hyperparameters.policy_learning_rate,
            alt_net_init=self.hyperparameters.alt_net_init,
            alt_net_learning_rate=self.hyperparameters.alt_net_learning_rate,
            target_entropy=self.hyperparameters.target_entropy,
        ))

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.score = 0.0
    
    def when_episode_starts(self, initial_observation, episode_index):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        self.previous_action = None
        self.previous_observation = None
        self.episode_index = episode_index
        # self.save_model()    
        
    # this may not be used
    def when_action_needed(self, observation, reward):
        """
        returns the action
        """
        # if not first decision
        if self.previous_action != None:
            # I'm not sure why reward is being divided by 10 here
            self.memory.put((self.previous_observation, self.previous_action, reward/10.0, observation, False))
            self.score += reward
        
        action, log_prob = self.policy(torch.from_numpy(observation).float())
        self.previous_action = action.item()
        self.previous_observation = observation
        
        # I'm not sure why this is a list or why its multiplying the action by two
        return [2.0*self.previous_action]
    
    def when_episode_ends(self, final_observation, reward, episode_index):
        # save the final observation
        # I'm not sure why reward is being divided by 10 here
        self.memory.put((self.previous_observation, self.previous_action, reward/10.0, final_observation, True))
        self.score += reward
        
        # training
        for _ in range(self.hyperparameters.minibatch_training_iterations):
            mini_batch = self.memory.sample(self.hyperparameters.batch_size)
            td_target = calc_target(self.policy, self.q1_target, self.q2_target, mini_batch, self.hyperparameters.discount_rate)
            self.q1.train_net(td_target, mini_batch)
            self.q2.train_net(td_target, mini_batch)
            entropy = self.policy.train_net(self.q1, self.q2, mini_batch)
            self.q1.soft_update(self.q1_target)
            self.q2.soft_update(self.q2_target)
        
        # logging
        if self.episode_index % self.print_interval == 0 and self.episode_index != 0:
            average_score = self.score/self.print_interval
            alpha = self.policy.log_alpha.exp()
            print(f"# of episode :{self.episode_index}, avg score : {average_score:.1f} alpha:{alpha:.4f}")
            self.score = 0.0
        
    
    def when_should_clean(self):
        """
        only called once, and should save checkpoints and cleanup any logging info
        """
        return

def test_sac():
    env = gym.make('Pendulum-v0')
    mr_bond = Agent()
    for n_epi in range(10000):
        observation = env.reset()
        reward = None
        done = False
        mr_bond.when_episode_starts(observation, n_epi)
        while not done:
            action = mr_bond.when_action_needed(observation, reward)
            observation, reward, done, info = env.step(action)
        mr_bond.when_episode_ends(observation, reward, n_epi)
                
    env.close()

if __name__ == '__main__':
    test_sac()