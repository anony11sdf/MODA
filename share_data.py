import torch
import prepare_data
from tcn_model import TCNModel
import numpy as np
import pickle
import time

hyperparameters = {
    'number_transition': 1,
    'output_size': 1,
    'tcn_layers': 4,
    'kernel_size': 3,
    'batch_size': 10,
    'aim_driver': 17,
    'slide_length': 2,
    'move_size': 1000,  
    'filter_out': -1, 
    'used_driver_size': 20,
}
transition_size = 252  
input_size = 252 + 127 * hyperparameters['number_transition']



def split_trajectory(trajectory, num_transitions):
    transitions = []
    for i in range(num_transitions):
        start_idx = i * 127  
        end_idx = start_idx + 252
        transition = trajectory[start_idx:end_idx]
        transitions.append(transition)  
    return transitions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCNModel(input_size, hyperparameters['output_size'], hyperparameters['tcn_layers'], hyperparameters['kernel_size']).to(device)


state_dict = torch.load('...')
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

model.eval()


st = time.time()


anchor_list = prepare_data.get_anchor(hyperparameters['aim_driver'], hyperparameters['number_transition'], hyperparameters['filter_out'], hyperparameters['used_driver_size'], hyperparameters['slide_length'], move_size=2)     # Implement this function

et = time.time()
execution_time = et - st


negative_list = prepare_data.get_negative(hyperparameters['aim_driver'], hyperparameters['number_transition'], hyperparameters['filter_out'], hyperparameters['used_driver_size'], hyperparameters['slide_length'], hyperparameters['move_size'])  # Implement this function




st = time.time()

anchor_embeddings = []
for anchor in anchor_list:

    anchor_tensor = torch.tensor(anchor, dtype=torch.float32).view(1, 1, -1).to(device)
    with torch.no_grad():
        anchor_embedding = model(anchor_tensor)
    anchor_embeddings.append(anchor_embedding.cpu().numpy())

average_anchor_embedding = np.mean(anchor_embeddings, axis=0)
x = np.mean([np.linalg.norm(embedding - average_anchor_embedding) for embedding in anchor_embeddings])


subtraj_buffer = []
for negative in negative_list:

    negative_tensor = torch.tensor(negative, dtype=torch.float32).view(1, 1, -1).to(device)
    with torch.no_grad():
        negative_embedding = model(negative_tensor).cpu().numpy()
    if np.linalg.norm(negative_embedding - average_anchor_embedding) <= x:
        subtraj_buffer.append(negative)


final_buffer = []
for trajectory in subtraj_buffer:
    transitions = split_trajectory(trajectory, hyperparameters['number_transition']) 
    for transition in transitions:

        final_buffer.append(transition)


with open('', 'rb') as file:
    no_share_buffer = pickle.load(file)
combined_buffer = no_share_buffer + final_buffer



with open('', 'wb') as file:
    pickle.dump(combined_buffer, file)

et = time.time()
execution_time = et - st
