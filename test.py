import random
import numpy as np
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

class OptionNet(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(OptionNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3)  
        self.fc1 = nn.Linear(16 * 3 * 3, 64)  
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 使用2x2的池化核
        x = x.view(-1, 16 * 3 * 3)  # 将特征张量展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ActionNet(nn.Module):
    def __init__(self, input_channels, output_dim, extra_option_dim):
        super(ActionNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3)
        self.fc1 = nn.Linear(16 * 3 * 3 + extra_option_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.extra_option_dim = extra_option_dim

    def forward(self, x, option):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 3 * 3)
        extra_option = option.view(-1, self.extra_option_dim)
        x = torch.cat((x, extra_option), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
input_tensor = torch.rand(8, 8, 3)
input_channels = 3
option_dim = 3
action_dim = 7
batch_size = 32

# Create an instance of the network
option_net = OptionNet(input_channels, option_dim)
action_net = ActionNet(input_channels, action_dim, option_dim)

# Assuming input is your input tensor
input_tensor = input_tensor.expand(batch_size, -1, -1, -1)
output = option_net(input_tensor)
probs = torch.nn.functional.softmax(output, dim=1)
option_dist = torch.distributions.Categorical(probs)
option = torch.argmax(option_dist.probs)
option = torch.tensor(option, dtype=torch.int64)
option = F.one_hot(option, num_classes = option_dim)

probs = action_net(input_tensor, option)
probs = torch.nn.functional.softmax(probs, dim=1)
action_dist = torch.distributions.Categorical(probs)
action = torch.argmax(action_dist.probs)




