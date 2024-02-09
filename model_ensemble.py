import torch
import torch.nn as nn


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
    
    

class DynamicModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size):
        super(DynamicModel, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.ensemble_size = ensemble_size


        self.ensemble_models = nn.ModuleList([SubModel_state(state_size + action_size, state_size) for _ in range(ensemble_size)])
        self.reward_models = nn.ModuleList([SubModel_reward(state_size + action_size, reward_size) for _ in range(ensemble_size)])

    def forward(self, state, action):
             
        state = state.to(dtype=torch.float32)  
        action = action.to(dtype=torch.float32) 
        
        predicted_states = [model(torch.cat([state, action], dim=1)) for model in self.ensemble_models]
        predicted_rewards = [model(torch.cat([state, action], dim=1)) for model in self.reward_models]

        return predicted_states, predicted_rewards    
    
