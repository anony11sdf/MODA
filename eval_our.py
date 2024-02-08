import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pickle
from torch.distributions import Normal
from torch.distributions import Categorical
import sys
from our_MDP import SimpleDynamicModel, Discriminator, CustomSACEnv_eval


sys.path.append('/home/xinbo/project1/evaluation/ensemble_model')
from model_ensemble import DynamicModel

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SAC Actor模型
class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound): ######离散的action，action_bound无所谓，任意值
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.logits = nn.Linear(256, action_dim)  # 使用 logits 来表示离散动作
        self.action_bound = action_bound


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)
        #print(f"Logits stats: max {logits.max().item()}, min {logits.min().item()}, mean {logits.mean().item()}")
        return logits

    def sample(self, state):
        logits = self.forward(state)
        action_dist = Categorical(logits=logits)  # 直接使用logits
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob



    # 选择动作的函数
    def select_action(self, state):
        logits = self.forward(state)
        action_dist = Categorical(logits=logits)  # 直接使用logits
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action



# 测试策略的函数
def test_policy_model(dynamic_model, policy_network, initial_state, stop_step, device):
    state = initial_state.unsqueeze(0).to(device)
    total_reward = 0
    effective_step_count = 0

    while effective_step_count < stop_step:
        action = policy_network.select_action(state)
        
        #print("action ",action, type(action))
        
        action_tensor =action.unsqueeze(1).float().to(device)

        predicted_states, predicted_rewards = dynamic_model(state, action_tensor)
        next_state = torch.mean(torch.stack(predicted_states), dim=0).squeeze()
        reward = torch.mean(torch.stack(predicted_rewards), dim=0).item()

        if reward == 200:
            break  # 跳过这一步，不算作有效步骤

        print(f"Step {effective_step_count + 1}: Reward = {reward}")
        total_reward += reward
        state = next_state.unsqueeze(0)
        effective_step_count += 1  # 有效步骤计数增加

    print(f"Total Reward after {effective_step_count} effective steps: {total_reward}")
    print(f"Average Reward after {effective_step_count} effective steps: {total_reward / stop_step}")
    return total_reward / stop_step





ensemble_size = 5
lr=0.002
batch_size=32
state_size = 125
action_size = 1
reward_size = 1
stop_step = 20
final = 0
# 加载训练好的模型

dynamic_model = DynamicModel(state_size, action_size, reward_size, ensemble_size).to(device)
dynamic_model.load_state_dict(torch.load('/home/xinbo/project1/evaluation/ensemble_model/dynamic_model_epoch_4999.pth'))
dynamic_model.eval()



policy_network = SACActor(state_dim=125, action_dim=19, action_bound=18).to(device)
policy_network.load_state_dict(torch.load('/home/xinbo/project1/RL/our_method/actor_model_17_model1.pth'))
policy_network.eval()
##### get initial_state
with open('/home/xinbo/project1/data/full_buffer_20driver.pkl', 'rb') as file:
    full_buffer = pickle.load(file)
random_transition = random.choice(full_buffer)
#random_transition = full_buffer[0]
initial_state = torch.tensor(random_transition[:125], dtype=torch.float32)

#rdm_idx = [115426, 116182, 102957, 10367, 227359, 38310, 138967, 244207, 37867, 3801]

episode_num = 10
# Test the policy model
for i in range(episode_num):
    
    random_transition = random.choice(full_buffer)
    #random_transition = full_buffer[rdm_idx[i]]
    #random_transition = full_buffer[0]
    initial_state = torch.tensor(random_transition[:125], dtype=torch.float32)
    final = final + test_policy_model(dynamic_model, policy_network, initial_state, stop_step, device)

print("final ",final/episode_num)










