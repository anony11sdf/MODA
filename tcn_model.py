import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TCNModel(nn.Module):

    def __init__(self, input_size, output_size, tcn_layers, kernel_size=3):
        super(TCNModel, self).__init__()
        self.tcn_layers = tcn_layers
        # 注意这里的 in_channels 设置为 1，数据只有一个通道
        self.tcn = self.build_tcn(1, output_size, tcn_layers, kernel_size)

    
    def build_tcn(self, input_size, output_size, num_layers, kernel_size):
        layers = []
        in_channels = input_size  # 将in_channels设置为input_size
        for _ in range(num_layers):
            layers += [
                weight_norm(nn.Conv1d(in_channels, output_size, kernel_size, padding=(kernel_size - 1) // 2)),
                nn.ReLU(inplace=True)
            ]
            in_channels = output_size
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.tcn(x)
        # Global average pooling to get an embedding vector
        x = torch.mean(x, dim=2)
        return x
