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


device = torch.device(...)

# SAC Actor
class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound): 
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.logits = nn.Linear(256, action_dim)  
        self.action_bound = action_bound


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)
        return logits

    def sample(self, state):
        logits = self.forward(state)
        action_dist = Categorical(logits=logits) 
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SACCritic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q1 = nn.Linear(256, 1)
        
        # Q2
        self.fc3 = nn.Linear(state_dim + action_dim, 256)
        self.fc4 = nn.Linear(256, 256)
        self.q2 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim = 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)

        q2 = F.relu(self.fc3(sa))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        return q1, q2


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        
        

        reward = torch.FloatTensor([reward]).to(device)
        
        done = 1 if done else 0  
        
        done = torch.IntTensor([done]).to(device) 
        
        
        
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

# train
def train_sac(env, actor_model, critic_model, target_critic_model, episodes, batch_size, gamma, tau, alpha):
    optimizer_actor = optim.Adam(actor_model.parameters(), lr=1e-3)
    optimizer_critic = optim.Adam(critic_model.parameters(), lr=1e-3)
    
    actor_losses = []
    critic_losses = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        print("Eposide:  ", episode)

        while True:
            
            
            
            action, log_prob = actor_model.sample(state)

            action_mapped = int(round(action.item()))  
            
            next_state, reward, done = env.step(action)
            


            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
               
                print(dones.dtype)
                print(device)
              
                states = torch.tensor(states, dtype=torch.float32).to(device)
                
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

                dones = dones.to(device)
                
                
                with torch.no_grad():
                    next_actions, next_log_probs = actor_model.sample(next_states)

                    
                    next_log_probs = next_log_probs.unsqueeze(1)
                    next_actions_one_hot = F.one_hot(next_actions, num_classes=19).float()
                    next_actions_one_hot = next_actions_one_hot.to(device)

                    target_Q1, target_Q2 = target_critic_model(next_states, next_actions_one_hot)
                    

                    
                    target_Q_min = torch.min(target_Q1, target_Q2) - alpha * next_log_probs
                    
                   
                    target_Q_values = rewards + (1 - dones) * gamma * target_Q_min

                
                
                action_one_hot = F.one_hot(actions, num_classes=19).float()
                action_one_hot = action_one_hot.to(device)
                
                current_Q1, current_Q2 = critic_model(states, action_one_hot) 
                

                 
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
                
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())                
                            

                for name, param in actor_model.named_parameters():
                    if param.grad is not None:
                        pass

                                
                optimizer_actor.zero_grad()
                actor_loss.backward()


                torch.nn.utils.clip_grad_norm_(actor_model.parameters(), max_norm=1.0)
                optimizer_actor.step()



                break
            

    torch.save(...)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


episodes = 20000  
batch_size = 64  
gamma = 0.99  
tau = 0.005  
alpha = 0.2  
buffer_capacity = 300000  
state_size = ..
action_size = ..
action_dim = ...
reward_size = ..
transition_size = 2 * state_size + action_size + reward_siez


replay_buffer = ReplayBuffer(capacity=buffer_capacity)



dynamic_model_path = '...'
discriminator_model_path = '...'



full_buffer = pickle.load(file)

dynamic_model = SimpleDynamicModel(state_size = state_size, action_size=1, reward_size=1, ensemble_size=1)
dynamic_model.load_state_dict(torch.load(dynamic_model_path))
dynamic_model.to(device)
dynamic_model.eval() 


discriminator_model = Discriminator(input_size=transition_size)  
discriminator_model.load_state_dict(torch.load(discriminator_model_path))
discriminator_model.to(device)
discriminator_model.eval()  


env = CustomSACEnv(dynamic_model=dynamic_model, discriminator_model=discriminator_model, full_buffer = full_buffer, device=device)


actor_model = SACActor(state_dim=state_size, action_dim , action_bound= ..).to(device)
critic_model = SACCritic(state_dim=state_size, action_dim ).to(device)


target_critic_model = SACCritic(state_dim=state_size, action_dim).to(device)


target_critic_model.load_state_dict(critic_model.state_dict())


train_sac(env, actor_model, critic_model, target_critic_model, episodes, batch_size, gamma, tau, alpha)




