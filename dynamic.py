import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import matplotlib.pyplot as plt


num_epochs=2000

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')


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
    def __init__(self, state_size, action_size, reward_size, ensemble_size=1):
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
        
 

class TransitionDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        transition = self.transitions[idx]
        state, action, reward, next_state = (
            torch.tensor(transition[:125], dtype=torch.float32),
            torch.tensor(transition[125], dtype=torch.float32).view(1),
            torch.tensor(transition[126], dtype=torch.float32).view(1),
            torch.tensor(transition[127:], dtype=torch.float32)
        )
        return torch.cat([state, action]), torch.cat([next_state, reward])

def train_simple_dynamic_model(model, train_dataloader, optimizer, num_epochs=2000):
    criterion = nn.MSELoss()
    model.train()
    losses = []

    for epoch in range(num_epochs):
        
        total_loss = 0
        
        for state_action, next_state_reward in train_dataloader:
            state_action = state_action.to(device)
            next_state = next_state_reward[:, :-1].to(device)  
            reward = next_state_reward[:, -1].to(device)  

            predicted_states, predicted_rewards = model(state_action[:, :125], state_action[:, 125:])
            
           
            state_loss = sum(criterion(pred_state, next_state) for pred_state in predicted_states)
            reward_loss = sum(criterion(pred_reward, reward.unsqueeze(1)) for pred_reward in predicted_rewards)
            loss = state_loss + reward_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()

        average_loss = total_loss / len(train_dataloader)
        losses.append(average_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')


    
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()
    plt.savefig('simple_dynamic.png')
    return model


with open('...', 'rb') as f:
    full_transition_buffer = pickle.load(f)


train_dataset = TransitionDataset(full_transition_buffer)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)



# Model initialization
state_size = 125
action_size = 1
reward_size = 1  

lr = 0.1
simple_dynamic_model = SimpleDynamicModel(state_size, action_size, reward_size).to(device)

# Optimizer
optimizer = optim.Adam(simple_dynamic_model.parameters(), lr)

# Train the model
trained_model = train_simple_dynamic_model(simple_dynamic_model, train_dataloader, optimizer, num_epochs)

# Save the trained model
torch.save(trained_model.state_dict(), '.....')
