import torch
import numpy as np
import gym
from gym import spaces
import torch.nn as nn
import math
import random
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import matplotlib.pyplot as plt

# 假设device已经定义
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

##########################################################################
##########################################################################
# 定义一个子模型
class SubModel_state(nn.Module):
    def __init__(self, input_size, output_size):
        super(SubModel_state, self).__init__()
        self.fc1 = nn.Linear(input_size, 64, dtype=torch.float32)
        #self.fc2 = nn.Linear(64, 64, dtype=torch.float32)
        self.fc2 = nn.Linear(64, output_size, dtype=torch.float32)

    def forward(self, x):
        
        x = x.to(dtype=torch.float32)  # 确保dtype相同
        
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = self.fc2(x)
        

        
        return x
    
    
    # 定义一个子模型
class SubModel_reward(nn.Module):
    def __init__(self, input_size, output_size):
        super(SubModel_reward, self).__init__()
        self.fc1 = nn.Linear(input_size, 32, dtype=torch.float32)
        #self.fc2 = nn.Linear(64, 64, dtype=torch.float32)
        self.fc2 = nn.Linear(32, output_size, dtype=torch.float32)

    def forward(self, x):
        
        x = x.to(dtype=torch.float32)  # 确保dtype相同
        
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = self.fc2(x)
        
        x = torch.clamp(x, min=-100, max=200)  # 限制输出范围
        
        return x
    
    
# 动态模型类
class SimpleDynamicModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size):
        super(SimpleDynamicModel, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.ensemble_size = ensemble_size

        # 创建多个子模型
        self.ensemble_models = nn.ModuleList([SubModel_state(state_size + action_size, state_size) for _ in range(ensemble_size)])
        self.reward_models = nn.ModuleList([SubModel_reward(state_size + action_size, reward_size) for _ in range(ensemble_size)])

    def forward(self, state, action):
        # 前向传播，对每个子模型进行计算
        
        
        combined_input = torch.cat([state, action], dim=1)  # 组合状态和动作
        predicted_states = [model(combined_input) for model in self.ensemble_models]
        predicted_rewards = [model(combined_input) for model in self.reward_models]
        return predicted_states, predicted_rewards

##########################################################################
##########################################################################



class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


##########################################################################
##########################################################################

class CustomSACEnv(gym.Env):
    """
    自定义环境，集成动态模型和判别器，用于SAC训练。
    """
    def __init__(self, dynamic_model, discriminator_model, state_size=125, action_size=1, full_buffer=None, device=None, discriminator_on = True):
        super(CustomSACEnv, self).__init__()
        self.dynamic_model = dynamic_model
        self.discriminator_model = discriminator_model
        self.state_size = state_size
        self.action_size = action_size
        self.full_buffer = full_buffer
        self.device = device
        self.discriminator_on = discriminator_on
        
        # 定义动作空间和状态空间的大小
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
        
        # 初始化状态
        self.state = torch.randn(state_size, dtype=torch.float32).to(self.device)

        # 创建 halt_state，全部元素为0
        self.halt_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)

        self.halt_reward = -100.0
        
    def step(self, action):
        # 将numpy数组的动作转换为torch张量
        
        action = torch.tensor(action, dtype=torch.float32).view(1, -1).to(self.device)  # 使action成为二维张量
        state = self.state.view(1, -1).to(self.device)
        
        
        
        
        # 使用动态模型预测下一个状态和奖励
        with torch.no_grad():
            #print("action ", action)
            #print(self.state.unsqueeze(0).shape)
            predicted_states, predicted_rewards = self.dynamic_model(self.state.unsqueeze(0), action)
            
        # 添加以下代码来打印动态模型的输出
        #print("predicted_states:", len(predicted_states))
        #print("predicted_rewards:", predicted_rewards)
        
        # 选择第一个模型的输出（如果有多个模型的情况）
        #next_state = predicted_states[0].squeeze(0)
        next_state = torch.mean(torch.stack(predicted_states), dim=0).squeeze()
        #reward = predicted_rewards[0].squeeze(0)
        reward = torch.mean(torch.stack(predicted_rewards), dim=0).squeeze(0)
        #print("reward ",reward.shape)
        #print("next_state",next_state.shape)
        
        # 构造完整的转移向量(s, a, r, s')
        
        
        transition = torch.cat([state.squeeze(0), action.squeeze(0), reward, next_state], dim=0).unsqueeze(0)
        
        if self.discriminator_on == True:
            discriminator_output = self.discriminator_model(transition)
        
        # 如果判别器输出小于0.8，将奖励设置为极大的负值
        #print("discriminator_out ",discriminator_output)
        
        # if discriminator_output.item() == 0.0:
        #     print("end episode")
        #     done = True
        #     print('done1 ', done)
        #     self.reset()
        #     print('done2 ',done)
        #     return return_state, return_reward, done, info
            #reward = torch.tensor([10.0], device=device)
        
        # 更新状态
        self.state = next_state.squeeze(0)
        
        # 这个环境不会自然结束（done始终为False），除非特定条件触发
        done = False
        
        # 可选的额外信息
        #info = {"discriminator_output": discriminator_output.item()}
        
        return_state = self.state 

        # 对于奖励，由于它是标量值，直接使用.item()获取值即可，不需要额外操作
        return_reward = reward.item()

        # 返回更新的状态、奖励、是否结束标志和额外信息
        
        if self.discriminator_on == True and discriminator_output.item() == 0.0 :
            print("end episode")
            done = True
            return_state = self.halt_state
            return_reward = self.halt_reward
            
        if state.max().item() > 1e8 or state.min().item() < -1e8:
            print("end episode big")
            done = True
            return_state = self.halt_state        
            return_reward = self.halt_reward
            
            
        if math.isnan(state.max().item()) or math.isnan(state.min().item()):
            print("end episode nan")
            done = True
            return_state = self.halt_state
            return_reward = self.halt_reward    
            
        # if return_reward > 100.0 or return_reward < 0.0:
        #     return_reward = -5.0
              
            
        print("return_state:", return_state.shape)
        #print("return_reward:", return_reward)            
              
        return return_state, return_reward, done
    
    

    def reset(self):
        # Load the full buffer from the pickle file
        if self.full_buffer is None:
            raise ValueError("full_buffer is not provided")
        
        # Randomly select a transition and take the first 125 dimensions as the initial state
        random_transition = random.choice(self.full_buffer)
        initial_state = torch.tensor(random_transition[:125], dtype=torch.float32).to(self.device)
        
        # Update the internal state
        #self.state = initial_state.to(self.device)
        
        # Return the initial state as a numpy array for compatibility
        #print("ini_state ", initial_state)
        return initial_state

    def render(self, mode='human'):
        pass  # 这个环境没有可视化的界面
    
    
    
##########################################################################
##########################################################################



class CustomSACEnv_eval(gym.Env):
    """
    自定义环境，集成动态模型和判别器，用于SAC训练。
    """
    def __init__(self, dynamic_model,  state_size=125, action_size=1, full_buffer=None, device=None, discriminator_on = False):
        super(CustomSACEnv_eval, self).__init__()
        self.dynamic_model = dynamic_model
        self.state_size = state_size
        self.action_size = action_size
        self.full_buffer = full_buffer
        self.device = device
        self.discriminator_on = discriminator_on
        
        # 定义动作空间和状态空间的大小
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
        
        # 初始化状态
        self.state = torch.randn(state_size, dtype=torch.float32).to(self.device)

        # 创建 halt_state，全部元素为0
        self.halt_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)

        self.halt_reward = -100.0
        
    def step(self, action):
        # 将numpy数组的动作转换为torch张量
        
        action = torch.tensor(action, dtype=torch.float32).view(1, -1).to(self.device)  # 使action成为二维张量
        state = self.state.view(1, -1).to(self.device)
        
        
        
        
        # 使用动态模型预测下一个状态和奖励
        with torch.no_grad():
            predicted_states, predicted_rewards = self.dynamic_model(self.state.unsqueeze(0), action)
            
        # 添加以下代码来打印动态模型的输出
        #print("predicted_states:", predicted_states)
        #print("predicted_rewards:", predicted_rewards)
        
        # 选择第一个模型的输出（如果有多个模型的情况）
        next_state = predicted_states[0].squeeze(0)
        reward = predicted_rewards[0].squeeze(0)
        
        # 构造完整的转移向量(s, a, r, s')
        transition = torch.cat([state.squeeze(0), action.squeeze(0), reward, next_state], dim=0).unsqueeze(0)
        
        #if self.discriminator_on == True:
            #discriminator_output = self.discriminator_model(transition)
        
        # 如果判别器输出小于0.8，将奖励设置为极大的负值
        #print("discriminator_out ",discriminator_output)
        
        # if discriminator_output.item() == 0.0:
        #     print("end episode")
        #     done = True
        #     print('done1 ', done)
        #     self.reset()
        #     print('done2 ',done)
        #     return return_state, return_reward, done, info
            #reward = torch.tensor([10.0], device=device)
        
        # 更新状态
        self.state = next_state.squeeze(0)
        
        # 这个环境不会自然结束（done始终为False），除非特定条件触发
        done = False
        
        # 可选的额外信息
        #info = {"discriminator_output": discriminator_output.item()}
        
        return_state = self.state 

        # 对于奖励，由于它是标量值，直接使用.item()获取值即可，不需要额外操作
        return_reward = reward.item()

        # 返回更新的状态、奖励、是否结束标志和额外信息
        

        if state.max().item() > 1e8 or state.min().item() < -1e8:
            print("end episode big")
            done = True
            return_state = self.halt_state        
            return_reward = self.halt_reward            

            
            
        if math.isnan(state.max().item()) or math.isnan(state.min().item()):
            print("end episode nan")
            done = True
            return_state = self.halt_state
            return_reward = self.halt_reward      
            
        #print("return_state:", return_state)
        #print("return_reward:", return_reward)            
              
        return return_state, return_reward, done
    
    

    def reset(self):
        # Load the full buffer from the pickle file
        if self.full_buffer is None:
            raise ValueError("full_buffer is not provided")
        
        # Randomly select a transition and take the first 125 dimensions as the initial state
        random_transition = random.choice(self.full_buffer)
        initial_state = torch.tensor(random_transition[:125], dtype=torch.float32).to(self.device)
        
        # Update the internal state
        #self.state = initial_state.to(self.device)
        
        # Return the initial state as a numpy array for compatibility
        print("ini_state ", initial_state)
        return initial_state

    def render(self, mode='human'):
        pass  # 这个环境没有可视化的界面
    
    
    
    
    



