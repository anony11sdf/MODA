import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle


# 定义生成器和判别器




# 定义生成器和鉴别器

# class Generator(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, output_size),
#             nn.Tanh()  # 输出层使用 Tanh 以输出介于 -1 和 1 之间的值
#         )
#         self.apply(self._init_weights)

#     def forward(self, x):
#         return self.model(x)
    
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.kaiming_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

# class Discriminator(nn.Module):
#     def __init__(self, input_size):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#         self.apply(self._init_weights)

#     def forward(self, x):
#         return self.model(x)
    
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.kaiming_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)








class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)