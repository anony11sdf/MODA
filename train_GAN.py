import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import GPUtil
import matplotlib.pyplot as plt 
from GAN_model import Generator, Discriminator


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



class TransitionDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)


    def __getitem__(self, idx):

        transition = torch.tensor(self.transitions[idx], dtype=torch.float32)
        return transition, transition  


input_size = 100  
output_size = 252  
batch_size = 32
epochs = 5000


with open('', 'rb') as file:
    buffer = pickle.load(file)

train_dataset = TransitionDataset(buffer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


generator = Generator(input_size, output_size).to(device)

discriminator = Discriminator(output_size).to(device)




gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0005)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)



gen_losses = []
dis_losses = []



for epoch in range(epochs):
    for real_data, _ in train_dataloader:
        real_data = real_data.to(device)
        batch_size = real_data.size(0)


        dis_optimizer.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        decision_real = discriminator(real_data)
        loss_real = nn.BCELoss()(decision_real, labels_real)

        noise = torch.randn(batch_size, input_size).to(device)
        generated_data = generator(noise).detach()
        labels_fake = torch.zeros(batch_size, 1).to(device)
        decision_fake = discriminator(generated_data)
        loss_fake = nn.BCELoss()(decision_fake, labels_fake)

        dis_loss = (loss_real + loss_fake) / 2
        dis_loss.backward()
        dis_optimizer.step()


        gen_optimizer.zero_grad()
        labels_gen = torch.ones(batch_size, 1).to(device)
        generated_data = generator(noise)
        decision_gen = discriminator(generated_data)
        gen_loss = nn.BCELoss()(decision_gen, labels_gen)

        gen_loss.backward()
        gen_optimizer.step()
        

    gen_losses.append(gen_loss.item())
    dis_losses.append(dis_loss.item())




torch.save(generator.state_dict(), '')
torch.save(discriminator.state_dict(), '')




