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

# 检查是否有可用的 CUDA 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化 TCN 模型并将其移动到 GPU 上
model = TCNModel(input_size, hyperparameters['output_size'], hyperparameters['tcn_layers'], hyperparameters['kernel_size']).to(device)

# 使用 DistributedDataParallel 包装模型，将模型在多个 GPU 上进行并行化
#model = DistributedDataParallel(model)

# 如果有多个 GPU 可用，使用 DataParallel 包装模型

if torch.cuda.device_count() > 1:
    print("使用多个 GPU...")
    model = nn.DataParallel(model)

# 定义 triplet margin loss 函数
def triplet_margin_loss(anchor, positive, negative, margin):
    d_positive = torch.norm(anchor - positive, dim=1) * 10  # 计算 anchor 和 positive 之间的距离
    d_negative = torch.norm(anchor - negative, dim=1) * 10  # 计算 anchor 和 negative 之间的距离
    
    
    print(d_positive, ' ',d_negative)
    
    loss = torch.relu(d_positive - d_negative + margin)  # 计算 triplet margin loss
    return loss.mean()  # 返回平均损失

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])

# 训练循环
losses = []

for epoch in range(hyperparameters['num_epochs']):
    total_loss = 0.0
    for batch_data in dataloader:
        optimizer.zero_grad()  # 清零梯度
        
        anchor, positive, negative = batch_data[0]  # 从数据中获取 anchor、positive 和 negative    
        
        anchor = anchor.unsqueeze(1)  # 改变形状为 [batch_size, 1, input_size]
        positive = positive.unsqueeze(1)
        negative = negative.unsqueeze(1)
            
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        outputs_anchor = model(anchor)  # 前向传播，获取特征表示
        outputs_positive = model(positive)
        outputs_negative = model(negative)
        loss = triplet_margin_loss(outputs_anchor, outputs_positive, outputs_negative, hyperparameters['margin'])  # 计算 triplet margin loss
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        total_loss += loss.item()
        
    # 计算平均损失并将其添加到列表中
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)

    # 打印每个 epoch 的损失

    print(f'Epoch [{epoch+1}/{hyperparameters["num_epochs"]}], Loss: {total_loss:.4f}')

# 保存训练好的模型
torch.save(model.state_dict(), 'tcn_triplet_model.pth')

print('训练完成并保存模型。')


# 设置图的大小，以确保足够的空间容纳图例
plt.figure(figsize=(20, 8))  # 适当调整 width 和 height 的值


# 绘制损失图表
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')

# 提取超参数名称和值，并组合成一个字符串
hyperparameters_str = ', '.join([f'{key}={value}' for key, value in hyperparameters.items()])

# 将超参数的名称和值作为图例显示在图的一角
plt.legend([hyperparameters_str], loc='upper right')


# 构建包含所有超参数的标题


plt.title('TCN loss')

plt.savefig('loss.png')
plt.show()


print(len(losses))

for key, value in hyperparameters.items():
    print(key,' ',value)
