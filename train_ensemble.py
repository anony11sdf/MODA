import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model_ensemble import DynamicModel, SubModel_state, SubModel_reward
import matplotlib.pyplot as plt
import os
import time




Epoch = 10000
ensemble_size = 5
lr=0.002
batch_size=64

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")



class TransitionDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        transition = self.transitions[idx]
        state, action, reward, next_state = (
            torch.tensor(transition[:125]),
            torch.tensor(transition[125]).unsqueeze(0),
            torch.tensor(transition[126]).unsqueeze(0),
            torch.tensor(transition[127:])
        )
        return torch.cat([state, action]), torch.cat([next_state, reward])



def train_dynamic_model(model, train_dataloader, optimizer, num_epochs=Epoch):
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(num_epochs):
        
        st = time.time()
        
        
        for batch in train_dataloader:
            state_action, next_state_reward = batch

            state_action = state_action.to(device).to(torch.float32)
            next_state_reward = next_state_reward.to(device).to(torch.float32)


          
            predicted_states, predicted_rewards = model(state_action[:, :125], state_action[:, 125:])

            
            state_loss = sum(criterion(predicted_state, next_state_reward[:, :125]) for predicted_state in predicted_states)
            reward_loss = sum(criterion(predicted_reward, next_state_reward[:, 125:]) for predicted_reward in predicted_rewards)
            loss = state_loss + reward_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            


        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
        
        et = time.time()
        execution_time = et - st
        print(f"batch时间：{execution_time:.2f}秒")
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0: 
            torch.save(...)
        
    torch.save( )
    


if __name__ == "__main__":

    state_size = 125
    action_size = 1
    reward_size = 1


    dynamic_model = DynamicModel(state_size, action_size, reward_size, ensemble_size)

    dynamic_model = dynamic_model.to(device) 
    

    
    
    optimizer = torch.optim.Adam(dynamic_model.parameters(), lr)

    with open('/home/xinbo/project1/data/full_buffer.pkl', 'rb') as f:
        full_transition_buffer = pickle.load(f)
        
    print(type(full_transition_buffer),len(full_transition_buffer),type(full_transition_buffer[0]),len(full_transition_buffer[0]))


    train_dataset = TransitionDataset(full_transition_buffer)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)


    train_dynamic_model(dynamic_model, train_dataloader, optimizer)
