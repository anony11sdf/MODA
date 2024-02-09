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



class SubModel_state(nn.Module):
    def __init__(self, input_size, output_size):
        super(SubModel_state, self).__init__()
        self.fc1 = nn.Linear(input_size, 64, dtype=torch.float32)
        self.fc2 = nn.Linear(64, output_size, dtype=torch.float32)

    def forward(self, x):
        
        x = x.to(dtype=torch.float32)  
        
        x = torch.relu(self.fc1(x))

        x = self.fc2(x)
        

        
        return x
    

class SubModel_reward(nn.Module):
    def __init__(self, input_size, output_size):
        super(SubModel_reward, self).__init__()
        self.fc1 = nn.Linear(input_size, 32, dtype=torch.float32)

        self.fc2 = nn.Linear(32, output_size, dtype=torch.float32)

    def forward(self, x):
        
        x = x.to(dtype=torch.float32)
        
        x = torch.relu(self.fc1(x))

        x = self.fc2(x)
        
        x = torch.clamp(x, min=-100, max=200)  
        
        return x
    
    

class SimpleDynamicModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size):
        super(SimpleDynamicModel, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.ensemble_size = ensemble_size

        self.ensemble_models = nn.ModuleList([SubModel_state(state_size + action_size, state_size) for _ in range(ensemble_size)])
        self.reward_models = nn.ModuleList([SubModel_reward(state_size + action_size, reward_size) for _ in range(ensemble_size)])

    def forward(self, state, action):

        combined_input = torch.cat([state, action], dim=1) 
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

    def __init__(self, dynamic_model, discriminator_model, state_size=125, action_size=1, full_buffer=None, device=None, discriminator_on = True):
        super(CustomSACEnv, self).__init__()
        self.dynamic_model = dynamic_model
        self.discriminator_model = discriminator_model
        self.state_size = state_size
        self.action_size = action_size
        self.full_buffer = full_buffer
        self.device = device
        self.discriminator_on = discriminator_on
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)
        

        self.state = torch.randn(state_size, dtype=torch.float32).to(self.device)


        self.halt_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)

        self.halt_reward = -100.0
        
    def step(self, action):

        
        action = torch.tensor(action, dtype=torch.float32).view(1, -1).to(self.device) 
        state = self.state.view(1, -1).to(self.device)
        
        
        

        with torch.no_grad():


            predicted_states, predicted_rewards = self.dynamic_model(self.state.unsqueeze(0), action)

        next_state = torch.mean(torch.stack(predicted_states), dim=0).squeeze()
        reward = torch.mean(torch.stack(predicted_rewards), dim=0).squeeze(0)

        
        transition = torch.cat([state.squeeze(0), action.squeeze(0), reward, next_state], dim=0).unsqueeze(0)
        
        if self.discriminator_on == True:
            discriminator_output = self.discriminator_model(transition)
        

        self.state = next_state.squeeze(0)
        
        done = False

        
        return_state = self.state 

        return_reward = reward.item()


        
        if self.discriminator_on == True and discriminator_output.item() == 0.0 :
            print("end episode")
            done = True
            return_state = self.halt_state
            return_reward = self.halt_reward
            
            
            
        if math.isnan(state.max().item()) or math.isnan(state.min().item()):
            print("end episode nan")
            done = True
            return_state = self.halt_state
            return_reward = self.halt_reward    
            

        print("return_state:", return_state.shape)         
              
        return return_state, return_reward, done
    
    

    def reset(self):
        # Load the full buffer from the pickle file
        if self.full_buffer is None:
            raise ValueError("full_buffer is not provided")
        
        # Randomly select a transition and take the first 125 dimensions as the initial state
        random_transition = random.choice(self.full_buffer)
        initial_state = torch.tensor(random_transition[:125], dtype=torch.float32).to(self.device)

        return initial_state

    def render(self, mode='human'):
        pass  
    
    
    
##########################################################################
##########################################################################




