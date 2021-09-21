import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random
from super_map import LazyDict

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, action, reward, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([action])
            r_lst.append([reward])
            s_prime_lst.append(s_prime)
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
        s, _, _, _, _ = mini_batch
        action, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,action), q2(s,action)
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
        s, action, reward, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, action) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.hyperparameters.tau) + param.data * self.hyperparameters.tau)

def calc_target(pi, q1, q2, mini_batch, discount_rate):
    s, action, reward, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob= pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
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
            policy_learning_rate  = 0.0005,
            critic_learning_rate  = 0.001,
            alt_net_init          = 0.01,
            alt_net_learning_rate = 0.001,  # for automated alpha update
            discount_rate         = 0.98,
            batch_size            = 32,
            buffer_limit          = 50000,
            tau                   = 0.01, # for target network soft update
            target_entropy        = -1.0, # for automated alpha update
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
        
        
    # this may not be used
    def decide(self, observation, reward, is_last_timestep):
        """
        returns the action
        """
        # if not first decision
        if self.previous_action != None:
            self.memory.put((self.previous_observation, self.previous_action, reward/10.0, observation, is_last_timestep))
            self.score += reward
        
        action, log_prob = self.policy(torch.from_numpy(observation).float())
        self.previous_action = action.item()
        self.previous_observation = observation
        
        # I'm not sure why this is a list or why its multiplying the action by two
        return [2.0*self.previous_action]
    
    
    def on_episode_start(self):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        self.previous_action = None
        self.previous_observation = None
        # self.save_model()
    
    def on_clean_up(self):
        """
        only called once, and should save checkpoints and cleanup any logging info
        """
        return
        
    def load_model(self):
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        
    def save_model(self):
        # self.show("saving model")
        # torch.save(self.model.state_dict(), self.checkpoint_path)
        pass
        
    def train(self):
        print("not yet implemented")

    def test(self):
        print("not yet implemented")

def main():
    env = gym.make('Pendulum-v0')
    mr_bond = Agent()
    for n_epi in range(10000):
        observation = env.reset()
        reward = None
        done = False
        mr_bond.on_episode_start()
        while not done:
            action = mr_bond.decide(observation, reward, done)
            observation, reward, done, info = env.step(action)
                
        if mr_bond.memory.size()>1000:
            for i in range(20):
                mini_batch = mr_bond.memory.sample(mr_bond.hyperparameters.batch_size)
                td_target = calc_target(mr_bond.policy, mr_bond.q1_target, mr_bond.q2_target, mini_batch, mr_bond.hyperparameters.discount_rate)
                mr_bond.q1.train_net(td_target, mini_batch)
                mr_bond.q2.train_net(td_target, mini_batch)
                entropy = mr_bond.policy.train_net(mr_bond.q1, mr_bond.q2, mini_batch)
                mr_bond.q1.soft_update(mr_bond.q1_target)
                mr_bond.q2.soft_update(mr_bond.q2_target)
                
        if n_epi%mr_bond.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, mr_bond.score/mr_bond.print_interval, mr_bond.policy.log_alpha.exp()))
            mr_bond.score = 0.0

    env.close()

if __name__ == '__main__':
    main()