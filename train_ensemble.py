import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model_ensemble import DynamicModel, SubModel_state, SubModel_reward
import matplotlib.pyplot as plt
import os
import time

#print(os.environ.get('CUDA_VISIBLE_DEVICES'))  # 查看当前设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # 设置为你想使用的GPU编号


Epoch = 10000
ensemble_size = 5
lr=0.002
batch_size=64
# 检查是否有可用的 CUDA 设备
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


# 用于训练的示例数据集类
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


# 训练动态模型的函数
def train_dynamic_model(model, train_dataloader, optimizer, num_epochs=Epoch):
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(num_epochs):
        
        st = time.time() #####记录每个epoch时间
        
        
        for batch in train_dataloader:
            state_action, next_state_reward = batch

            state_action = state_action.to(device).to(torch.float32)
            next_state_reward = next_state_reward.to(device).to(torch.float32)


            # 前向传播
            predicted_states, predicted_rewards = model(state_action[:, :125], state_action[:, 125:])

            # 计算损失
            state_loss = sum(criterion(predicted_state, next_state_reward[:, :125]) for predicted_state in predicted_states)
            reward_loss = sum(criterion(predicted_reward, next_state_reward[:, 125:]) for predicted_reward in predicted_rewards)
            loss = state_loss + reward_loss

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            


        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
        
        et = time.time()
        execution_time = et - st
        print(f"batch时间：{execution_time:.2f}秒")
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:  # 在每100个epoch结束后保存模型
            torch.save(model.state_dict(), f'dynamic_model_epoch_{epoch}.pth')
        
    torch.save(model.state_dict(), 'dynamic_model_final.pth')
    
    # 绘制损失的折线图
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_ensemble.png')


if __name__ == "__main__":
    # 定义模型和优化器
    state_size = 125
    action_size = 1
    reward_size = 1


    dynamic_model = DynamicModel(state_size, action_size, reward_size, ensemble_size)
    
    #if torch.cuda.device_count() > 100:
        #print("使用4个 GPU!")
        #dynamic_model = nn.DataParallel(dynamic_model)
        # 如果你希望使用指定的GPU，比如使用前4个GPU，可以这样做：
        #dynamic_model = nn.DataParallel(dynamic_model, device_ids=[0,1,2,3])
    
    
    
    
    dynamic_model = dynamic_model.to(device)  # 将模型移动到GPU
    
    
    # if torch.cuda.device_count() > 1:
    #     print("使用多个GPU...")
    #     device_ids = [7,6,5,4,3,2,1]  # 你希望使用的 GPU 设备的列表
    #     dynamic_model = nn.DataParallel(dynamic_model, device_ids=device_ids)
    
    
    optimizer = torch.optim.Adam(dynamic_model.parameters(), lr)

    # 导入数据集
    with open('/home/xinbo/project1/data/full_buffer.pkl', 'rb') as f:
        full_transition_buffer = pickle.load(f)
        
    print(type(full_transition_buffer),len(full_transition_buffer),type(full_transition_buffer[0]),len(full_transition_buffer[0]))

    # 将数据集转换成Dataset和DataLoader
    train_dataset = TransitionDataset(full_transition_buffer)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    # 训练动态模型
    train_dynamic_model(dynamic_model, train_dataloader, optimizer)