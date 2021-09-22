import gym
import math
import random
import numpy as np
import matplotlib # FIXME: plt
import matplotlib.pyplot as plt # FIXME: plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# debugging tools
# - what is affecting the models decision (time-adjusted)
# - distance from one model weights to another model weights
# - variability in learning (rewards bouncing around)

# local 
from tools.all_tools import PATHS, product, device, layer_output_shapes

class Agent:
    def __init__(self, action_space=None, train=True, **config):
        """
        arguments:
            action_space: needs to be discrete
        """
        self.config = config
        self.action_space = action_space
        self.wants_to_quit = False
        self.show = lambda *args, **kwargs: print(*args, **kwargs) if config.get("suppress_output", False) else None
        
    
    def when_episode_starts(self, initial_observation, episode_index):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        
        return
        
    # this may not be used
    def when_action_needed(self, observation, reward):
        """
        returns an action from the action space
        """
        return
    
    def when_should_clean(self):
        """
        only called once, and should save checkpoints and cleanup any logging info
        """
        return


class Model():
    class ReplayMemory(object):
        transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.memory.append(Model.ReplayMemory.transition(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    class DQN(nn.Module):

        def __init__(self, input_shape, output_shape):
            """
            input_shape = (channels, height, width)
            """
            super(DQN, self).__init__()
            
            # upgrade shape to 3D if 2D
            if len(input_shape) == 2: input_shape = (1, *input_shape)
            
            channels, height, width  = input_shape
            self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, stride=2)
            self.bn1   = nn.BatchNorm2d(self.conv1.out_channels)
            self.conv2 = nn.Conv2d(self.bn1.num_features, 32, kernel_size=5, stride=2)
            self.bn2   = nn.BatchNorm2d(self.conv2.out_channels)
            self.conv3 = nn.Conv2d(self.bn2.num_features, 32, kernel_size=5, stride=2)
            self.bn3   = nn.BatchNorm2d(self.conv3.out_channels)
            
            layers = [ self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, ]
            shape_of_last_layer = layer_output_shapes(layers, input_shape)[-1]
            self.head = nn.Linear(product(*shape_of_last_layer), product(output_shape))

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            """
            x:
                either an input image or batch of images
                (batch_size, channels, height, width)
            """
            # convert images into batches of 1 if needed
            if len(x.shape) == 3: x = torch.reshape(x, (1, x.shape))
                
            x = x.to(device)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
        

# TODO: map train, test, and predict into the agent (maybe have different start options)

# 
# other code
# 

# FIXME: no env
env = gym.make('CartPole-v0').unwrapped

# FIXME: matplotlib
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()




resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),
    T.ToTensor()
])

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    
    # FIXME: no shortcuts/chopping
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    # what is x_threshold? 
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    cart_location = int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    
    
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


# FIXME: plt
env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')
plt.title('Example extracted screen')
plt.show()



# 
# training 
# 
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())