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
        return transition, transition  # 第二个 transition 作为占位符，实际不使用

# 超参数
input_size = 100  # 用于生成器的随机噪声向量大小
output_size = 252  # 生成的transition向量大小
batch_size = 32
epochs = 5000

# 加载数据集
with open('/home/xinbo/project1/data/UDS_full_buffer_20driver.pkl', 'rb') as file:
    buffer = pickle.load(file)

# 创建数据加载器
train_dataset = TransitionDataset(buffer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化生成器和判别器
generator = Generator(input_size, output_size).to(device)

discriminator = Discriminator(output_size).to(device)



# 定义生成器和判别器的优化器
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0005)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)


# 用于记录损失的列表
gen_losses = []
dis_losses = []


# 训练循环
for epoch in range(epochs):
    for real_data, _ in train_dataloader:
        real_data = real_data.to(device)
        batch_size = real_data.size(0)

        # 训练判别器
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

        # 训练生成器
        gen_optimizer.zero_grad()
        labels_gen = torch.ones(batch_size, 1).to(device)
        generated_data = generator(noise)
        decision_gen = discriminator(generated_data)
        gen_loss = nn.BCELoss()(decision_gen, labels_gen)

        gen_loss.backward()
        gen_optimizer.step()
        

    gen_losses.append(gen_loss.item())
    dis_losses.append(dis_loss.item())

    print(f'Epoch [{epoch}/{epochs}], 生成器损失: {gen_loss.item():.4f}, 判别器损失: {dis_loss.item():.4f}')
    if (epoch + 1) % 1000 == 0:  # 在每100个epoch结束后保存模型
        torch.save(generator.state_dict(), f'GAN_generator_UDS_epoch_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'GAN_discriminator_UDS_epoch_{epoch+1}.pth')


torch.save(generator.state_dict(), 'generator_model17_UDS.pth')
torch.save(discriminator.state_dict(), 'discriminator_model17_UDS.pth')

# 绘制生成器和判别器的损失折线图
plt.plot(gen_losses, label='Generator Loss')
plt.plot(dis_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('gan_loss_plot.png')
plt.show()


#########使用判别器时需要阈值
