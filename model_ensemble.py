import torch
import torch.nn as nn

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
class DynamicModel(nn.Module):
    def __init__(self, state_size, action_size, reward_size, ensemble_size):
        super(DynamicModel, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.reward_size = reward_size
        self.ensemble_size = ensemble_size

        # 创建多个子模型
        self.ensemble_models = nn.ModuleList([SubModel_state(state_size + action_size, state_size) for _ in range(ensemble_size)])
        self.reward_models = nn.ModuleList([SubModel_reward(state_size + action_size, reward_size) for _ in range(ensemble_size)])

    def forward(self, state, action):
        # 前向传播，对每个子模型进行计算
        
        state = state.to(dtype=torch.float32)  # 确保dtype相同
        action = action.to(dtype=torch.float32)  # 确保dtype相同
        
        predicted_states = [model(torch.cat([state, action], dim=1)) for model in self.ensemble_models]
        predicted_rewards = [model(torch.cat([state, action], dim=1)) for model in self.reward_models]

        return predicted_states, predicted_rewards    
    
    

# # 定义整体的动态模型
# class DynamicModel(nn.Module):
#     def __init__(self, state_size, action_size, reward_size, ensemble_size=5):
#         super(DynamicModel, self).__init__()

#         self.state_size = state_size
#         self.action_size = action_size
#         self.reward_size = reward_size
#         self.ensemble_size = ensemble_size

#         # 创建一个由多个子模型组成的集合
#         self.ensemble_models = nn.ModuleList([SubModel(state_size + action_size, state_size + reward_size) for _ in range(ensemble_size)])

#     def forward(self, state, action):
#         # 将状态和动作连接在一起
#         x = torch.cat([state, action], dim=1)

#         # 通过集合中的每个子模型进行前向传播
#         ensemble_outputs = [model(x) for model in self.ensemble_models]

#         # 将输出拆分为预测的状态和奖励
#         predicted_states = torch.stack([output[:, :self.state_size] for output in ensemble_outputs], dim=1)
#         predicted_rewards = torch.stack([output[:, self.state_size:] for output in ensemble_outputs], dim=1)

#         return predicted_states, predicted_rewards
