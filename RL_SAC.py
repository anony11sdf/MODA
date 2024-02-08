import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pickle
from torch.distributions import Normal
from torch.distributions import Categorical
import matplotlib.pyplot as plt 
from our_MDP import SimpleDynamicModel, Discriminator, CustomSACEnv

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

# SAC Critic模型
class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SACCritic, self).__init__()
        # Q1网络
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q1 = nn.Linear(256, 1)
        # Q2网络
        self.fc3 = nn.Linear(state_dim + action_dim, 256)
        self.fc4 = nn.Linear(256, 256)
        self.q2 = nn.Linear(256, 1)

    def forward(self, state, action):
        #action = action.unsqueeze(0)  # 在action上添加一个额外的维度
        sa = torch.cat([state, action], dim = 1)

        # Q1的前向传播
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        # Q2的前向传播
        q2 = F.relu(self.fc3(sa))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        return q1, q2

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        
        
        # 将所有数据从NumPy数组转换为PyTorch张量
        #state = torch.FloatTensor(state).to(device).clone()
        #action = torch.LongTensor(action).to(device)  # 使用 LongTensor 存储整数动作
        reward = torch.FloatTensor([reward]).to(device)
        #next_state = torch.FloatTensor(next_state).to(device)
        #done = torch.FloatTensor(done).to(device).unsqueeze(1)
        
        done = 1 if done else 0  # 将布尔值转换为整数
        
        done = torch.IntTensor([done]).to(device)  # 转换为整数张量
        
        
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# SAC训练函数
def train_sac(env, actor_model, critic_model, target_critic_model, episodes, batch_size, gamma, tau, alpha):
    optimizer_actor = optim.Adam(actor_model.parameters(), lr=1e-3)
    optimizer_critic = optim.Adam(critic_model.parameters(), lr=1e-3)
    
    actor_losses = []
    critic_losses = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        print("Eposide:  ", episode)
        #state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
        while True:
            
            
            #print(f"State tensor stats: max {state.max().item()}, min {state.min().item()}, mean {state.mean().item()}")
            
            action, log_prob = actor_model.sample(state)
            #action_np = action.cpu().detach().numpy()
            #print('tag0')           
            # 将离散动作映射到你的动作空间范围（0到18）
            action_mapped = int(round(action.item()))  # 四舍五入到最接近的整数
            
            next_state, reward, done = env.step(action)
            
            # print('state: ', type(state))   
            # print('next_state: ', type(next_state))   
            # print('done: ', done)

            #print('tag1')
            
            # print('reward ', type(reward),' ', reward)
            # print('action', action)
            # print('action_mapped', action_mapped)

            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
               
                print(dones.dtype)
                print(device)
              
                states = torch.tensor(states, dtype=torch.float32).to(device)
                #actions = torch.tensor(actions, dtype=torch.int64).to(device)  
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                
                #print(f"Next state stats: max {np.max(next_state)}, min {np.min(next_state)}, mean {np.mean(next_state)}")
                # print(f"Reward: {reward}")
                # print(f"Action: {action}")
                
            # 添加以下两行打印语句以检查维度
                #print(next_states.shape)
                #print(next_actions_tensor.shape)
                
                #dones = torch.FloatTensor(dones).to(device).unsqueeze(1)
                dones = dones.to(device)
                
                
                with torch.no_grad():
                    next_actions, next_log_probs = actor_model.sample(next_states)
                    #next_actions = torch.tensor(next_actions, dtype=torch.float32).view(-1, 1).to(device)
                    #next_actions_np = next_actions.cpu().numpy()  # 将下一步动作转换为 NumPy 数组
                    #next_actions_tensor = torch.FloatTensor(next_actions_np).unsqueeze(1).to(device)  # 将 NumPy 数组转换为张量
                    
                    
                    next_log_probs = next_log_probs.unsqueeze(1)
                    next_actions_one_hot = F.one_hot(next_actions, num_classes=19).float()
                    next_actions_one_hot = next_actions_one_hot.to(device)
                    #print('next_states',next_states.shape)
                    #print('next_actions_one_hot',next_actions_one_hot.shape,' ',next_actions_one_hot[0])        
                    
                    
                    
                    target_Q1, target_Q2 = target_critic_model(next_states, next_actions_one_hot)
                    
                    #print("target_Q1 ",target_Q1.shape)
                    #print("target_Q2 ",target_Q2.shape)     
                    #print("next_log_probs ",next_log_probs.shape)               
                    #tmp = torch.min(target_Q1, target_Q2)
                    #print("tmp", tmp.shape)
                    
                    target_Q_min = torch.min(target_Q1, target_Q2) - alpha * next_log_probs
                    
                    #print("target_Q_min ",target_Q_min.shape)
                    
                    #print("dones ",dones.shape)
                    #print("rewards ",rewards.shape)                    
                    target_Q_values = rewards + (1 - dones) * gamma * target_Q_min

                
                
                action_one_hot = F.one_hot(actions, num_classes=19).float()
                action_one_hot = action_one_hot.to(device)
                
                current_Q1, current_Q2 = critic_model(states, action_one_hot) 
                
                #print("current_Q1 ", current_Q1.shape)
                #print("target_Q_values ", target_Q_values.shape)
                 
                critic_loss = F.mse_loss(current_Q1, target_Q_values) + F.mse_loss(current_Q2, target_Q_values)

                optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_norm=1.0)
                optimizer_critic.step()

                soft_update(target_critic_model, critic_model, tau)

                new_actions, log_probs = actor_model.sample(states)
                
                new_actions_one_hot = F.one_hot(new_actions, num_classes=19).float()
                new_actions_one_hot = next_actions_one_hot.to(device)                
                
                
                Q1_new, Q2_new = critic_model(states, new_actions_one_hot)
                Q_min_new = torch.min(Q1_new, Q2_new)
                actor_loss = (alpha * log_probs - Q_min_new).mean()
                
                
                # 计算critic_loss和actor_loss之后
                print(f"Critic loss: {critic_loss.item()}, Actor loss: {actor_loss.item()}")
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())                
                            
                # 在optimizer_critic.step()和optimizer_actor.step()之前，对actor_model的所有参数
                for name, param in actor_model.named_parameters():
                    if param.grad is not None:
                        print(f"Grad - {name}: max: {param.grad.max().item()}, min: {param.grad.min().item()}, mean: {param.grad.mean().item()}")

                                
                optimizer_actor.zero_grad()
                actor_loss.backward()


                torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
                optimizer_actor.step()

            if done:
                print("done  ",done)
                print(f"Episode {episode}: Total Reward: {episode_reward}")

                break
            
    # 在train_sac函数结束前添加以下代码来绘制并保存loss图像
    plt.figure(figsize=(12, 6))
    plt.plot(actor_losses, label='Actor Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Actor Loss Over Episodes')
    plt.legend()
    plt.savefig('actor_loss.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(critic_losses, label='Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Critic Loss Over Episodes')
    plt.legend()
    plt.savefig('critic_loss.png')
    plt.close()        
    
    
    
    torch.save(actor_model.state_dict(), 'actor_model_17_model1.pth')
    #torch.save(critic_model.state_dict(), 'path_to_save_critic_model.pth')
    #torch.save(target_critic_model.state_dict(), 'path_to_save_target_critic_model.pth')
            
            

# 软更新目标模型
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# Hyperparameters
episodes = 20000  # 训练的总episode数
batch_size = 64  # 批量大小
gamma = 0.99  # 折扣因子
tau = 0.005  # 目标网络的软更新系数
alpha = 0.2  # 熵正则化系数
buffer_capacity = 300000  # 经验回放缓冲区的容量

# 初始化经验回放缓冲区
replay_buffer = ReplayBuffer(capacity=buffer_capacity)

# 创建自定义环境

dynamic_model_path = '/home/xinbo/project1/models/GAN_MDP/aim_driver_17/model1/dynamic_epo10000/simple_dynamic_model_epoch_4999.pth'
discriminator_model_path = '/home/xinbo/project1/models/GAN_MDP/aim_driver_17/model1/discriminator_model.pth'


file = open('/home/xinbo/project1/data/full_buffer_20driver.pkl', 'rb')
full_buffer = pickle.load(file)
# 加载动态模型
dynamic_model = SimpleDynamicModel(state_size=125, action_size=1, reward_size=1, ensemble_size=1)
dynamic_model.load_state_dict(torch.load(dynamic_model_path))
dynamic_model.to(device)
dynamic_model.eval()  # 设置为评估模式

# 加载判别器模型
discriminator_model = Discriminator(input_size=252)  # transition size
discriminator_model.load_state_dict(torch.load(discriminator_model_path))
discriminator_model.to(device)
discriminator_model.eval()  # 设置为评估模式

# 请确保你已经定义了 CustomSACEnv 类，它应该接受 dynamic_model 和 discriminator_model 作为参数
# 并实现 reset 和 step 方法
env = CustomSACEnv(dynamic_model=dynamic_model, discriminator_model=discriminator_model, full_buffer = full_buffer, device=device)

# 初始化SAC Actor和Critic模型
actor_model = SACActor(state_dim=125, action_dim=19, action_bound=18).to(device)
critic_model = SACCritic(state_dim=125, action_dim=19).to(device)

# 克隆Critic模型以创建目标Critic模型
target_critic_model = SACCritic(state_dim=125, action_dim=19).to(device)

# 初始化目标Critic模型的权重和Critic模型的权重相同
target_critic_model.load_state_dict(critic_model.state_dict())

# 调用train_sac
train_sac(env, actor_model, critic_model, target_critic_model, episodes, batch_size, gamma, tau, alpha)












# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import random
# import pickle
# from torch.distributions import Normal
# from torch.distributions import Categorical

# from our_MDP import SimpleDynamicModel, Discriminator, CustomSACEnv

# # 定义设备
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # SAC Actor模型
# class SACActor(nn.Module):
#     def __init__(self, state_dim, action_dim, action_bound): ######离散的action，action_bound无所谓，任意值
#         super(SACActor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.logits = nn.Linear(256, action_dim)  # 使用 logits 来表示离散动作
#         self.action_bound = action_bound


#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         logits = self.logits(x)
#         print(f"Logits stats: max {logits.max().item()}, min {logits.min().item()}, mean {logits.mean().item()}")
#         return logits

#     def sample(self, state):
#         logits = self.forward(state)
#         action_dist = Categorical(logits=logits)  # 直接使用logits
#         action = action_dist.sample()
#         log_prob = action_dist.log_prob(action)
#         return action, log_prob

# # SAC Critic模型
# class SACCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(SACCritic, self).__init__()
#         # Q1网络
#         self.fc1 = nn.Linear(state_dim + action_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.q1 = nn.Linear(256, 1)
#         # Q2网络
#         self.fc3 = nn.Linear(state_dim + action_dim, 256)
#         self.fc4 = nn.Linear(256, 256)
#         self.q2 = nn.Linear(256, 1)

#     def forward(self, state, action):
#         #action = action.unsqueeze(0)  # 在action上添加一个额外的维度
#         sa = torch.cat([state, action], dim = 1)

#         # Q1的前向传播
#         q1 = F.relu(self.fc1(sa))
#         q1 = F.relu(self.fc2(q1))
#         q1 = self.q1(q1)
#         # Q2的前向传播
#         q2 = F.relu(self.fc3(sa))
#         q2 = F.relu(self.fc4(q2))
#         q2 = self.q2(q2)
#         return q1, q2

# # 经验回放缓冲区
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = []
#         self.position = 0

#     def add(self, state, action, reward, next_state, done):
        
        
#         # 将所有数据从NumPy数组转换为PyTorch张量
#         #state = torch.FloatTensor(state).to(device).clone()
#         #action = torch.LongTensor(action).to(device)  # 使用 LongTensor 存储整数动作
#         reward = torch.FloatTensor([reward]).to(device)
#         #next_state = torch.FloatTensor(next_state).to(device)
#         #done = torch.FloatTensor(done).to(device).unsqueeze(1)
        
#         done = 1 if done else 0  # 将布尔值转换为整数
        
#         done = torch.IntTensor([done]).to(device)  # 转换为整数张量
        
        
        
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(None)
#         self.buffer[self.position] = (state, action, reward, next_state, done)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done = map(torch.stack, zip(*batch))
#         return state, action, reward, next_state, done

#     def __len__(self):
#         return len(self.buffer)

# # SAC训练函数
# def train_sac(env, actor_model, critic_model, target_critic_model, episodes, batch_size, gamma, tau, alpha):
#     optimizer_actor = optim.Adam(actor_model.parameters(), lr=1e-3)
#     optimizer_critic = optim.Adam(critic_model.parameters(), lr=1e-3)

#     for episode in range(episodes):
#         state = env.reset()
#         episode_reward = 0
        
#         print("Eposide:  ", episode)
#         #state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
#         while True:
            
            
#             print(f"State tensor stats: max {state.max().item()}, min {state.min().item()}, mean {state.mean().item()}")
            
#             action, log_prob = actor_model.sample(state)
#             #action_np = action.cpu().detach().numpy()
#             #print('tag0')           
#             # 将离散动作映射到你的动作空间范围（0到18）
#             action_mapped = int(round(action.item()))  # 四舍五入到最接近的整数
            
#             next_state, reward, done, _ = env.step(action)
            
#             print('state: ', type(state))   
#             print('next_state: ', type(next_state))   
#             print('done: ', done)

#             #print('tag1')
            
#             print('reward ', type(reward),' ', reward)
#             print('action', action)
#             print('action_mapped', action_mapped)

#             replay_buffer.add(state, action, reward, next_state, done)

#             state = next_state
#             episode_reward += reward

#             if len(replay_buffer) > batch_size:
#                 states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
               
#                 print(dones.dtype)
#                 print(device)
              
#                 states = torch.tensor(states, dtype=torch.float32).to(device)
#                 #actions = torch.tensor(actions, dtype=torch.int64).to(device)  
#                 rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
#                 next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                
#                 #print(f"Next state stats: max {np.max(next_state)}, min {np.min(next_state)}, mean {np.mean(next_state)}")
#                 print(f"Reward: {reward}")
#                 print(f"Action: {action}")
                
#             # 添加以下两行打印语句以检查维度
#                 #print(next_states.shape)
#                 #print(next_actions_tensor.shape)
                
#                 #dones = torch.FloatTensor(dones).to(device).unsqueeze(1)
#                 dones = dones.to(device)
                
                
#                 with torch.no_grad():
#                     next_actions, next_log_probs = actor_model.sample(next_states)
#                     #next_actions = torch.tensor(next_actions, dtype=torch.float32).view(-1, 1).to(device)
#                     #next_actions_np = next_actions.cpu().numpy()  # 将下一步动作转换为 NumPy 数组
#                     #next_actions_tensor = torch.FloatTensor(next_actions_np).unsqueeze(1).to(device)  # 将 NumPy 数组转换为张量
                    
                    
#                     next_log_probs = next_log_probs.unsqueeze(1)
#                     next_actions_one_hot = F.one_hot(next_actions, num_classes=19).float()
#                     next_actions_one_hot = next_actions_one_hot.to(device)
#                     print('next_states',next_states.shape)
#                     print('next_actions_one_hot',next_actions_one_hot.shape,' ',next_actions_one_hot[0])        
                    
                    
                    
#                     target_Q1, target_Q2 = target_critic_model(next_states, next_actions_one_hot)
                    
#                     print("target_Q1 ",target_Q1.shape)
#                     print("target_Q2 ",target_Q2.shape)     
#                     print("next_log_probs ",next_log_probs.shape)               
#                     tmp = torch.min(target_Q1, target_Q2)
#                     print("tmp", tmp.shape)
                    
#                     target_Q_min = torch.min(target_Q1, target_Q2) - alpha * next_log_probs
                    
#                     print("target_Q_min ",target_Q_min.shape)
                    
#                     print("dones ",dones.shape)
#                     print("rewards ",rewards.shape)                    
#                     target_Q_values = rewards + (1 - dones) * gamma * target_Q_min

                
                
#                 action_one_hot = F.one_hot(actions, num_classes=19).float()
#                 action_one_hot = action_one_hot.to(device)
                
#                 current_Q1, current_Q2 = critic_model(states, action_one_hot) 
                
#                 print("current_Q1 ", current_Q1.shape)
#                 print("target_Q_values ", target_Q_values.shape)
                 
#                 critic_loss = F.mse_loss(current_Q1, target_Q_values) + F.mse_loss(current_Q2, target_Q_values)

#                 optimizer_critic.zero_grad()
#                 critic_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(critic_model.parameters(), max_norm=1.0)
#                 optimizer_critic.step()

#                 soft_update(target_critic_model, critic_model, tau)

#                 new_actions, log_probs = actor_model.sample(states)
                
#                 new_actions_one_hot = F.one_hot(new_actions, num_classes=19).float()
#                 new_actions_one_hot = next_actions_one_hot.to(device)                
                
                
#                 Q1_new, Q2_new = critic_model(states, new_actions_one_hot)
#                 Q_min_new = torch.min(Q1_new, Q2_new)
#                 actor_loss = (alpha * log_probs - Q_min_new).mean()
                
                
#                 # 计算critic_loss和actor_loss之后
#                 print(f"Critic loss: {critic_loss.item()}, Actor loss: {actor_loss.item()}")
                
#                 # 在optimizer_critic.step()和optimizer_actor.step()之前，对actor_model的所有参数
#                 for name, param in actor_model.named_parameters():
#                     if param.grad is not None:
#                         print(f"Grad - {name}: max: {param.grad.max().item()}, min: {param.grad.min().item()}, mean: {param.grad.mean().item()}")

                                
#                 optimizer_actor.zero_grad()
#                 actor_loss.backward()


#                 torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
#                 optimizer_actor.step()

#             if done:
#                 print(f"Episode {episode}: Total Reward: {episode_reward}")
#                 break
            
            
#     torch.save(actor_model.state_dict(), 'path_to_save_actor_model.pth')
#     torch.save(critic_model.state_dict(), 'path_to_save_critic_model.pth')
#     torch.save(target_critic_model.state_dict(), 'path_to_save_target_critic_model.pth')
            
            

# # 软更新目标模型
# def soft_update(target, source, tau):
#     for target_param, param in zip(target.parameters(), source.parameters()):
#         target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# # Hyperparameters
# episodes = 10000  # 训练的总episode数
# batch_size = 64  # 批量大小
# gamma = 0.99  # 折扣因子
# tau = 0.005  # 目标网络的软更新系数
# alpha = 0.2  # 熵正则化系数
# buffer_capacity = 10000  # 经验回放缓冲区的容量

# # 初始化经验回放缓冲区
# replay_buffer = ReplayBuffer(capacity=buffer_capacity)

# # 创建自定义环境

# dynamic_model_path = '/home/xinbo/project1/evaluation/ensemble_model/dynamic_model_epoch_999.pth'
# discriminator_model_path = '/home/xinbo/project1/models/GAN_MDP/aim_driver_17/model1/discriminator_model.pth'


# file = open('/home/xinbo/project1/data/full_buffer_20driver.pkl', 'rb')
# full_buffer = pickle.load(file)
# # 加载动态模型
# dynamic_model = SimpleDynamicModel(state_size=125, action_size=1, reward_size=1, ensemble_size=5)
# dynamic_model.load_state_dict(torch.load(dynamic_model_path))
# dynamic_model.to(device)
# dynamic_model.eval()  # 设置为评估模式

# # 加载判别器模型
# discriminator_model = Discriminator(input_size=252)  # transition size
# discriminator_model.load_state_dict(torch.load(discriminator_model_path))
# discriminator_model.to(device)
# discriminator_model.eval()  # 设置为评估模式

# # 请确保你已经定义了 CustomSACEnv 类，它应该接受 dynamic_model 和 discriminator_model 作为参数
# # 并实现 reset 和 step 方法
# env = CustomSACEnv(dynamic_model=dynamic_model, discriminator_model=discriminator_model, full_buffer = full_buffer, device=device)

# # 初始化SAC Actor和Critic模型
# actor_model = SACActor(state_dim=125, action_dim=19, action_bound=18).to(device)
# critic_model = SACCritic(state_dim=125, action_dim=19).to(device)

# # 克隆Critic模型以创建目标Critic模型
# target_critic_model = SACCritic(state_dim=125, action_dim=19).to(device)

# # 初始化目标Critic模型的权重和Critic模型的权重相同
# target_critic_model.load_state_dict(critic_model.state_dict())

# # 调用train_sac
# train_sac(env, actor_model, critic_model, target_critic_model, episodes, batch_size, gamma, tau, alpha)