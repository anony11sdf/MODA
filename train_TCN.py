import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import torch.nn as nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tcn_model import TCNModel
import matplotlib.pyplot as plt 
import prepare_data


hyperparameters = {
    'number_transition': 1,
    'output_size': 1,  
    'tcn_layers': 4,  
    'kernel_size': 3,  
    'batch_size': 32,  
    'learning_rate': 0.001, 
    'num_epochs': 2000,  
    'margin': 2,
    'filter_out': 90,
    'used_driver_size': 20,
    'aim_driver': 17,
    'slide_length': 1,  
    'move_size': 1,
    
}



window_size = 252 + 127 * (hyperparameters['number_transition']-1)
input_size = 1  



tuple_list = prepare_data.get_triples(hyperparameters['aim_driver'], hyperparameters['number_transition'], hyperparameters['filter_out'], hyperparameters['used_driver_size'], hyperparameters['slide_length'],hyperparameters['move_size'])

sample_number = len(tuple_list)
print(tuple_list[100])

print("success prepare tuples, amount is ", sample_number)

batched_data = torch.stack([torch.stack(t) for t in tuple_list], dim=1)


dataset = TensorDataset(batched_data)
dataloader = DataLoader(dataset, batch_size=hyperparameters['batch_size'], shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = TCNModel(input_size, hyperparameters['output_size'], hyperparameters['tcn_layers'], hyperparameters['kernel_size']).to(device)



if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

def triplet_margin_loss(anchor, positive, negative, margin):
    d_positive = torch.norm(anchor - positive, dim=1) * 10  
    d_negative = torch.norm(anchor - negative, dim=1) * 10  
    
    
    print(d_positive, ' ',d_negative)
    
    loss = torch.relu(d_positive - d_negative + margin)
    return loss.mean()  


optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])


losses = []

for epoch in range(hyperparameters['num_epochs']):
    total_loss = 0.0
    for batch_data in dataloader:
        optimizer.zero_grad() 
        
        anchor, positive, negative = batch_data[0]    
        
        anchor = anchor.unsqueeze(1) 
        positive = positive.unsqueeze(1)
        negative = negative.unsqueeze(1)
            
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        outputs_anchor = model(anchor) 
        outputs_positive = model(positive)
        outputs_negative = model(negative)
        loss = triplet_margin_loss(outputs_anchor, outputs_positive, outputs_negative, hyperparameters['margin'])  
        loss.backward()  
        optimizer.step()  
        total_loss += loss.item()
        

    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)





torch.save(model.state_dict(), '')


