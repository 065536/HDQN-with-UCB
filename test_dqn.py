import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import *
import torch.nn.functional as F
import random
from torch.utils.tensorboard import SummaryWriter
import time
import datetime

import os

class ReplayMemory(object):
    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.Transition = Transition

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# class OptionNet(nn.Module):
#     def __init__(self, input_channels, output_dim):
#         super(OptionNet, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 16, 3)  
#         self.fc1 = nn.Linear(16 * 3 * 3, 64)  
#         self.fc2 = nn.Linear(64, output_dim)

#     def forward(self, x, mask):
#         x = x.permute(0, 3, 1, 2)
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)  # 使用2x2的池化核
#         x = x.reshape(-1, 16 * 3 * 3)  # 将特征张量展平
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         x -= (1 - mask) * 1e8
#         return x

class ActionNet(nn.Module):
    def __init__(self, input_channels, output_dim, extra_option_dim, dir_dim = 4):
        super(ActionNet, self).__init__()
        self.dir_dim = dir_dim
        self.conv1 = nn.Conv2d(input_channels, 16, 3)
        self.fc1 = nn.Linear(16 * 3 * 3 + extra_option_dim +self.dir_dim , 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.extra_option_dim = extra_option_dim
        

    def forward(self, x, option, mask, dir):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = x.reshape(-1, 16 * 3 * 3)
        extra_option = option.view(-1, self.extra_option_dim)
        x = torch.cat((x, extra_option, dir), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x -= (1 - mask) * 1e8
        return x

class FixOption():
    def __init__(self, agent, n_options = 3):
        self.n_options = n_options
        self.agent = agent

    def get_option(self, n_key, n_door_lock):
        #open the door
        if isinstance(self.agent.env.carrying, Key) and n_door_lock > 0:
            return 1
        
        elif n_key > 0:
            return 0
        
        else:
            return 2

class AIAgent:
    def __init__(self, env, state_channel, action_dim, option_dim, eps_action = 1, eps_option = 1, lr=0.001, gamma=0.99, capacity=10000, batch_size=32):
        # self.option_network = OptionNet(state_channel, option_dim)
        # self.option_target_network = OptionNet(state_channel, option_dim)
        self.action_network = ActionNet(state_channel, action_dim, option_dim)
        self.action_target_network = ActionNet(state_channel, action_dim, option_dim)
        self.action_optimizer = optim.Adam(self.action_network.parameters(), lr=lr)
        # self.option_optimizer = optim.Adam(self.option_network.parameters(), lr=lr)
        self.option_dim = option_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.total_step = 0
        self.eps_action = eps_action
        self.eps_option = eps_option
        self.env = env
        self.memory_action = ReplayMemory(capacity, Transition_action)
        # self.memory_option = ReplayMemory(capacity, Transition_option)
        self.batch_size = batch_size
        self.action_counts = torch.ones(1, action_dim, dtype=torch.float)
        self.option_counts = torch.ones(1, option_dim, dtype=torch.float)
        self.action_set = [i for i in range(action_dim)]
        self.option_set = [i for i in range(option_dim)]
        self.same_location_count = 0

    def train_action(self):
        if len(self.memory_action) < self.batch_size:
            return None, None
        transitions = self.memory_action.sample(self.batch_size)
        batch = Transition_action(*zip(*transitions))

        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        option_batch = batch.option
        option_batch = torch.stack(option_batch, dim=0)
        legal_action = torch.stack(batch.legal_action)
        dir_batch = batch.dir
        dir_batch = torch.stack(dir_batch)

        state_action_values = self.action_network(state_batch, option_batch, legal_action, dir_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(self.batch_size)
        next_state_values = self.action_target_network(non_final_next_states, option_batch, legal_action, dir_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        loss_action = loss
        self.action_optimizer.zero_grad()
        loss.backward()

        gradients = []
        for param in self.action_network.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        gradients = torch.cat(gradients)
        gradient_norm = torch.norm(gradients, 2)  # 2表示L2范数       
        self.action_optimizer.step()
        return loss_action, gradient_norm

    # def train_option(self):
    #     if len(self.memory_option) < self.batch_size:
    #         return None, None
    #     transitions = self.memory_option.sample(self.batch_size)
    #     batch = Transition_option(*zip(*transitions))
    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.end_state)), dtype=torch.bool)
    #     non_final_next_states = torch.stack([s for s in batch.end_state if s is not None])
    #     state_batch = torch.stack(batch.start_state)
    #     reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
    #     mask = torch.stack(batch.mask)

    #     option_values = self.option_network(state_batch, mask)
    #     next_option_values = torch.zeros(self.batch_size)
    #     next_option_values[non_final_mask] = self.option_target_network(non_final_next_states, mask).max(1)[0].detach()
    #     expected_option_values = (next_option_values * self.gamma) + reward_batch
    #     loss = nn.MSELoss()(option_values, expected_option_values.unsqueeze(1))
    #     loss_option = loss
    #     self.option_optimizer.zero_grad()
    #     loss.backward()

    #     gradients = []
    #     for param in self.option_network.parameters():
    #         if param.grad is not None:
    #             gradients.append(param.grad.view(-1))
    #     gradients = torch.cat(gradients)
    #     gradient_norm = torch.norm(gradients, 2)  # 2表示L2范数
    #     self.option_optimizer.step()
    #     print("loss_option", loss_option)
    #     print("gradient_norm", gradient_norm)
    #     return loss_option, gradient_norm

    def store_action_experience(self, state, option, action, next_state, reward, done, legal_action, dir):
        self.memory_action.push(state, option, action, next_state, reward, done, legal_action, dir)

    # def store_option_experience(self, start_state, option, step_length, end_state, reward, mask):
    #     self.memory_option.push(start_state, option, step_length, end_state, reward, mask)

    def update_action_target_networks(self):
        self.action_target_network.load_state_dict(self.action_network.state_dict())
       
    # def update_option_target_networks(self):
    #     self.option_target_network.load_state_dict(self.option_network.state_dict())

    # def select_action_ucb(self, state, option, c, steps_done):
    #     with torch.no_grad():
    #         action_values = self.action_network(state, option)
    #         ucb_values = action_values + c * torch.sqrt(math.log(steps_done + 1) / (self.action_counts + 1))  # 计算置信上界
    #         self.action_counts[0][action] += 1
    #         return ucb_values.max(1)[1].view(1, 1)

    # def select_option_ucb(self, state, c, steps_done):
    #     with torch.no_grad():
    #         action_values = self.option_network(state)
    #         ucb_values = action_values + c * torch.sqrt(math.log(steps_done + 1) / (self.option_counts + 1))  # 计算置信上界
    #         self.option_counts[0][option] += 1
    #         return ucb_values.max(1)[1].view(1, 1)
    
    def action_eps_greedy(self, state, option, legal_action, dir):
        if random.random() < self.eps_action:
            action = [random.random() for _ in range(self.action_dim)]
            action = torch.tensor(action)
            action *= legal_action
            probs = torch.nn.functional.softmax(action, dim=0)
            action_dist = torch.distributions.Categorical(probs)
            action = torch.argmax(action_dist.probs)

        else:
            batched_state = state.expand(self.batch_size, -1, -1, -1)
            batched_option = option.expand(self.batch_size, -1)
            batched_dir = dir.expand(self.batch_size, -1)
            probs = self.action_network(batched_state, batched_option, legal_action, batched_dir)
            probs = torch.nn.functional.softmax(probs, dim=1)
            probs = probs[0]
            action_dist = torch.distributions.Categorical(probs)
            action = torch.argmax(action_dist.probs)

        if env.step_count % 10 == 0:
            self.eps_action = self.eps_action * 0.99

        return action
    
    # def option_eps_greedy(self, state, mask):
    #     if random.random() < self.eps_action:
    #         option = [random.random() for _ in range(self.option_dim)]
    #         option = torch.tensor(option)
    #         option *= mask
    #         probs = torch.nn.functional.softmax(option, dim=0)
    #         option_dist = torch.distributions.Categorical(probs)
    #         option = torch.argmax(option_dist.probs)
    #     else:
    #         batched_state = state.expand(self.batch_size, -1, -1, -1)
    #         probs = self.option_network(batched_state, mask)
    #         probs = torch.nn.functional.softmax(probs, dim=1)
    #         probs = probs[0]
    #         option_dist = torch.distributions.Categorical(probs)
    #         option = torch.argmax(option_dist.probs)

    #     if env.step_count % 10 == 0:
    #         self.eps_option = self.eps_option*0.9
    #     return option
    
    def check_key_door(self, state):
        '''
        0: get key
        1: open the door
        2: go to goal
        '''

        n_key = 0
        n_door_lock = 0
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                #there exists a key
                if state[i, j, 0] == 5:
                    #get the key
                    n_key += 1
                
                #the door is not open
                elif state[i, j, 0] == 4 and state[i, j, 2] != 0:
                    n_door_lock += 1
        
        return n_key, n_door_lock
    
    # def get_option_mask(self, n_key, n_door_lock):
    #     mask = torch.ones(self.option_dim)
    #     mask_value = 1e-16
    #     if isinstance(self.env.carrying, Key) and n_door_lock > 0:
    #         #open the door
    #         mask[0] = mask_value
    #         mask[2] = mask_value
        
    #     elif n_key > 0:
    #         #get the key
    #         mask[1] = mask_value
    #         mask[2] = mask_value
        
    #     else:
    #         #go to goal
    #         mask[0] = mask_value
    #         mask[1] = mask_value
        
    #     return mask

    def get_legal_action(self):
        legal_action= torch.ones(self.action_dim)
        mask_value = 1e-16

        legal_action[6] = mask_value

        fwd_pos = self.env.front_pos
        fwd_cell = self.env.grid.get(*fwd_pos)

        #cannot move forward
        if fwd_cell == None or fwd_cell.can_overlap():
            pass
        else:
            legal_action[2] = mask_value
            
        #cannot pick up
        if (fwd_cell != None) and (fwd_cell.type == 'key') and fwd_cell.can_pickup() and (self.env.carrying is None):
            pass
        else:
            legal_action[3] = mask_value

        #cannot drop
        legal_action[4] = mask_value
        # if isinstance(self.env.carrying, Key) or (not self.env.carrying):
        #     legal_action[4] = mask_value
        # else:
        #     pass

        #cannot activate
        if fwd_cell != None and (fwd_cell.type == "door") and (not fwd_cell.is_open):
            pass
        else:
            legal_action[5] = mask_value
        
        return legal_action

    def check_option_done(self, option_num, n_door_lock, state, done):

        option_done = False
        #get the key
        if option_num == 0:
            if isinstance(self.env.carrying, Key):
                option_done =  True
        elif option_num == 1:
            _, n_door_lock_now = self.check_key_door(state)
            if n_door_lock_now < n_door_lock:
                option_done = True
        else:
            if done:
                option_done = True
        
        return option_done
        
if __name__ == "__main__":
    # 设置随机种子以复现结果
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    TARGET_UPDATE = 10
    GAMMA = 0.99
    MEMORY_SIZE = 10000

    project_name = "MiniGrid-DoorKey-8x8-v0"
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", project_name + "_" + current_time)

    # 如果GPU可用，则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 经验回放
    Transition_action = namedtuple('TransitionAction', ('state', 'option', 'action','next_state', 'reward', 'done', 'legal_action', 'dir'))
    # Transition_option = namedtuple('TransitionOption', ('start_state', 'option', 'step_length', 'end_state', 'reward', 'mask'))

    env_name = "MiniGrid-DoorKey-8x8-v0"
    # env_name = "MiniGrid-MultiRoom-N6-v0"
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env.max_steps = 10000

    current_directory = os.getcwd()
    logs_folder = os.path.join(current_directory, "logs")
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    writer = SummaryWriter(log_dir)

    # 初始化网络和优化器
    n_states = env.observation_space.shape[0] 
    n_actions = env.action_space.n
    n_options = 3 # 例如：拿钥匙、开门、去终点
    state_channel = 3
    n_direction = 4

    # 训练循环
    num_episodes = 1000
    while True:
        
        agent = AIAgent(env, state_channel, n_actions, n_options)
        fixOptionPolicy = FixOption(agent)
        for i_episode in range(num_episodes):
            print(f'training epoch: {i_episode}')
            start_state = env.reset()
            current_loc = env.agent_pos
            start_state = torch.tensor(start_state, device=device, dtype = torch.float)
            state = start_state
            # action_counts = torch.ones(1, n_actions, device=device, dtype=torch.float)
            # option_counts = torch.ones(1, n_options, device=device, dtype=torch.float)
            total_reward = 0.0
            done = False
            option_done = False
            n_key, n_door_lock = agent.check_key_door(state)

            # mask = agent.get_option_mask(n_key, n_door_lock)
            # option = agent.option_eps_greedy(start_state, mask)
            option_num = fixOptionPolicy.get_option(n_key, n_door_lock)
            print("option_num: ", option_num)
            option = torch.tensor(option_num, dtype=torch.int64)
            option = F.one_hot(option, num_classes = n_options)

            while not done:
                step = 0
                while (not done) and (not option_done):
                    agent.total_step += 1
                    env.render('human')
                    time.sleep(0.01)
                    step += 1
                    n_key_now, n_door_lock = agent.check_key_door(state)
                    legal_action = agent.get_legal_action()
                    dir = agent.env.agent_dir
                    dir = torch.tensor(dir, dtype=torch.int64)
                    dir = F.one_hot(dir, num_classes = n_direction)

                    action = agent.action_eps_greedy(state, option, legal_action, dir)

                    next_state, reward, done, _ = env.step(action)
                    now_loc = env.agent_pos
                    if np.array_equal(now_loc, current_loc):
                        agent.same_location_count += 1
                    else:
                        current_loc = now_loc
                        agent.same_location_count = 0
                    n_key_next, n_door_lock_next = agent.check_key_door(next_state)

                    if action == 5 and n_door_lock_next < n_door_lock: # 开门给正奖励
                        reward += 0.5

                    if n_key_next < n_key_now: # 拿到钥匙给奖励
                        reward += 0.25

                    #原地打转给负奖励
                    if agent.same_location_count >= 10:
                        reward -=0.01 
                    reward -= 0.005
                    next_state = torch.tensor(next_state, device=device, dtype=torch.float)
                    agent.store_action_experience(state, option, action, next_state, reward, done, legal_action, dir)

                    option_done = agent.check_option_done(option_num, n_door_lock, next_state, done)
                    loss_action, gradient_norm_action = agent.train_action()
                    # loss_option, gradient_norm_option = agent.train_option()
                    state = next_state
                    total_reward += reward
                    if step % TARGET_UPDATE == 0:
                        agent.update_action_target_networks()

                    writer.add_scalar('total_reward', total_reward, agent.total_step)
                    if loss_action is None:
                        pass
                    else:
                        writer.add_scalar('action_loss', loss_action, agent.total_step)
                    # if loss_option is None:
                    #     pass
                    # else:
                    #     writer.add_scalar('action_option', loss_option, agent.total_step)
                    if gradient_norm_action is None:
                        pass
                    else:
                        writer.add_scalar('gradient_norm_action', gradient_norm_action, agent.total_step)
                    # if gradient_norm_option is None:
                    #     pass
                    # else:
                    #     writer.add_scalar('gradient_norm_option', gradient_norm_option, agent.total_step)

                print(f"option {option_num} done. Use step {step}.")

                # agent.store_option_experience(start_state, option_num, step, state, reward, mask)

                print(f"-----------------option {option_num} done-----------------")

                if not done and option_done:
                    n_key, n_door_lock = agent.check_key_door(state)
                    # mask = agent.get_option_mask(n_key, n_door_lock)
                    # option = agent.option_eps_greedy(state, mask)
                    option = fixOptionPolicy.get_option(n_key, n_door_lock)
                    option_num = option
                    print("option_num: ", option_num)
                    option = torch.tensor(option)
                    option = F.one_hot(option, num_classes = n_options)
                    option_done = False
                    current_loc = env.agent_pos
                    agent.same_location_count = 0
            
            # if i_episode % TARGET_UPDATE == 0:
            #     agent.update_option_target_networks()
            
            print(f"Episode {i_episode}, Total Reward: {total_reward}")

        # torch.save(agent.option_network.state_dict(), 'option_model.pth')
        torch.save(agent.action_network.state_dict(), 'action_model.pth')
        print('done')
