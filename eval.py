import sys
import torch
import numpy as np
from model_ensemble import DynamicModel
import pickle
import random


# 添加CQL.py所在的目录到sys.path
sys.path.append('/home/xinbo/project1/baseline/CQL/')

# 现在可以导入CQL.py中的QNetwork类
from CQL import QNetwork

####ensemble parameter

ensemble_size = 5
lr=0.002
batch_size=32
state_size = 125
action_size = 1
reward_size = 1
stop_step = 20

###CQL parameter
state_dim = 125
action_dim = 19  # 动作0到18
hidden_dim = 256


#region 
# eval   CQL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cql_model = QNetwork(state_dim, action_dim, hidden_dim).to(device)
cql_model.load_state_dict(torch.load('/home/xinbo/project1/baseline/CQL/cql_model1_share_No17.pth'))
cql_model.eval()

def select_action(model, state):
    """
    根据当前状态选择动作。
    Args:
    - model (torch.nn.Module): CQL模型
    - state (torch.Tensor): 当前状态，一个125维的向量

    Returns:
    - action (int): 选择的动作
    """
    with torch.no_grad():
        # 将状态转换为模型所需的形式
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 获取Q值
        q_values = model(state)

        # 选择Q值最高的动作
        action = torch.argmax(q_values).item()

    return action


#endregion


# 加载训练好的模型

dynamic_model = DynamicModel(state_size, action_size, reward_size, ensemble_size).to(device)
dynamic_model.load_state_dict(torch.load('/home/xinbo/project1/evaluation/ensemble_model/dynamic_model_epoch_399.pth'))
dynamic_model.eval()




def test_rl_model(rl_model, initial_state, stop_step):
    state = initial_state.to(dtype=torch.float32).unsqueeze(0).to(device)

    total_reward = 0
    for step in range(stop_step):
        action = select_action(rl_model, state)  # 选择动作

        # 将 action 转换为 tensor 并确保它有正确的形状 [1, 1]
        action_tensor = torch.tensor([action], dtype=torch.float32).unsqueeze(0).to(device)

        # 连接 state 和 action_tensor
        state_action = torch.cat([state, action_tensor], dim=1)  # 注意这里的改变

        predicted_states, predicted_rewards = dynamic_model(state_action[:, :125], state_action[:, 125:])  # 预测
        next_state = torch.mean(torch.stack(predicted_states), dim=0).squeeze()
        reward = torch.mean(torch.stack(predicted_rewards), dim=0).item()

        print(f"Step {step + 1}: Reward = {reward}")
        total_reward += reward
        state = next_state.unsqueeze(0)  # 更新状态

    print(f"Total Reward after {stop_step} steps: {total_reward}")
    print(f"Average Reward after {stop_step} steps: {total_reward/stop_step}")
    return total_reward





##### get initial_state
with open('/home/xinbo/project1/data/full_buffer.pkl', 'rb') as file:
    full_buffer = pickle.load(file)
random_transition = random.choice(full_buffer)

initial_state = torch.tensor(random_transition[:125], dtype=torch.float32)



test_rl_model(cql_model, initial_state, stop_step)


