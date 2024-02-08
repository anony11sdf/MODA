import torch
import prepare_data
from tcn_model import TCNModel
import numpy as np
import pickle
import time
# 定义超参数
hyperparameters = {
    'number_transition': 1,
    'output_size': 1,
    'tcn_layers': 4,
    'kernel_size': 3,
    'batch_size': 10,
    'aim_driver': 17,
    'slide_length': 2,
    'move_size': 1000,   #  遍历所有，不限制move_size
    'filter_out': -1, #no filter
    'used_driver_size': 20,
}
transition_size = 252  # 每个transition的固定大小
input_size = 252 + 127 * hyperparameters['number_transition']



def split_trajectory(trajectory, num_transitions):
    transitions = []
    for i in range(num_transitions):
        start_idx = i * 127  # 每个transition重叠127维
        end_idx = start_idx + 252
        transition = trajectory[start_idx:end_idx]
        transitions.append(transition)  # 直接使用transition，不需要转换成list
    return transitions



# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCNModel(input_size, hyperparameters['output_size'], hyperparameters['tcn_layers'], hyperparameters['kernel_size']).to(device)

# 加载状态字典，并调整键以匹配模型期望的键
state_dict = torch.load('/home/xinbo/project1/models/TCN/aim_driver_17/model3/tcn_triplet_model.pth')
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

model.eval()


st = time.time()
# 获取negative和anchor数据




anchor_list = prepare_data.get_anchor(hyperparameters['aim_driver'], hyperparameters['number_transition'], hyperparameters['filter_out'], hyperparameters['used_driver_size'], hyperparameters['slide_length'], move_size=2)     # Implement this function

et = time.time()
execution_time = et - st
print(f"准备anchor时间：{execution_time:.2f}秒")

negative_list = prepare_data.get_negative(hyperparameters['aim_driver'], hyperparameters['number_transition'], hyperparameters['filter_out'], hyperparameters['used_driver_size'], hyperparameters['slide_length'], hyperparameters['move_size'])  # Implement this function

print('nega_finish')


st = time.time()
# 计算所有anchor的平均embedding
anchor_embeddings = []
for anchor in anchor_list:
    # 将anchor转换为正确的形状: [batch_size, 1, input_size]
    anchor_tensor = torch.tensor(anchor, dtype=torch.float32).view(1, 1, -1).to(device)
    with torch.no_grad():
        anchor_embedding = model(anchor_tensor)
    anchor_embeddings.append(anchor_embedding.cpu().numpy())

average_anchor_embedding = np.mean(anchor_embeddings, axis=0)
x = np.mean([np.linalg.norm(embedding - average_anchor_embedding) for embedding in anchor_embeddings])

# 处理negative list
subtraj_buffer = []
for negative in negative_list:
    # 将negative转换为正确的形状: [batch_size, 1, input_size]
    negative_tensor = torch.tensor(negative, dtype=torch.float32).view(1, 1, -1).to(device)
    with torch.no_grad():
        negative_embedding = model(negative_tensor).cpu().numpy()
    if np.linalg.norm(negative_embedding - average_anchor_embedding) <= x:
        subtraj_buffer.append(negative)

# 转换subtraj_buffer中的trajectory为transitions
final_buffer = []
for trajectory in subtraj_buffer:
    transitions = split_trajectory(trajectory, hyperparameters['number_transition'])  # Implement this function
    for transition in transitions:
        #if not any(np.array_equal(np.array(transition), np.array(existing_transition)) for existing_transition in final_buffer):
        final_buffer.append(transition)

# 读取并合并no_share_buffer
with open('/home/xinbo/project1/data/no_share_buffer_driver_No17.pkl', 'rb') as file:
    no_share_buffer = pickle.load(file)
combined_buffer = no_share_buffer + final_buffer

print("combined_buffer = no_share_buffer + final_buffer",len(combined_buffer),len(no_share_buffer),len(final_buffer))

# 保存最终的buffer
with open('/home/xinbo/project1/models/TCN/aim_driver_17/model3/buffer_V1.pkl', 'wb') as file:
    pickle.dump(combined_buffer, file)


et = time.time()
execution_time = et - st
print(f"sharing data 时间：{execution_time:.2f}秒")