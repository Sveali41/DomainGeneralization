import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import numpy as np

def compute_deconv_params(input_size, target_size):
    stride = int(np.ceil(target_size / input_size))
    padding = int(((input_size - 1) * stride + 3 - target_size) // 2)
    if padding < 0:
        padding = 0
    kernel_size = target_size - ((input_size - 1) * stride - 2 * padding)
    if padding < 0:
        padding = 0
    return kernel_size, stride, padding

class Generator(nn.Module):
    def __init__(self, z_shape, width, height, num_classes=6):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_shape, 32)
        
        # 使用BatchNorm进行规范化
        self.bn1 = nn.BatchNorm1d(32)
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=0)
        # ConvTranspose: Output size= (Input_size − 1) × Stride − 2 * Padding + Kernel_size
    def forward(self, z):
        z = z.view(z.size(0), -1)
        x = F.relu(self.bn1(self.fc(z)))  # 使用BatchNorm
        x = x.view(-1, 32, 1, 1)  # 重塑为卷积层输入格式
        x = F.relu(self.bn2(self.conv1(x)))  # 使用BatchNorm
        x = F.relu(self.bn3(self.conv2(x)))  # 使用BatchNorm
        x = self.conv3(x)  # 输出6x6大小的网格
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_channels, width, height):
        super(Discriminator, self).__init__()
        # Conv: Output size= (input_size + 2 * Padding - Kernel_size) / Stride + 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)  # 输入6x6
        # self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)  # 提取更多特征
        self.bn2 = nn.BatchNorm2d(32)
        
        self.fc = nn.Linear(32 * 2 * 2, 1)  # 压平后全连接层输出真假概率
        
        # self.dropout = nn.Dropout(0.3)  # Dropout 防止过拟合

    def forward(self, x):   
        if x.dim() == 3:
            # reshaped_input = x.view(-1)  
            # one_hot_encoded = F.one_hot(reshaped_input, 6)  
            # x = one_hot_encoded.view(x.shape[0], 6, 6, 6).permute(0, 3, 1, 2)  # Shape (32, 6, 8, 8)
            x = x.unsqueeze(1)

        x = x.float()
        x = F.leaky_relu(self.conv1(x), 0.2)  # LeakyReLU激活
        # x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)  # LeakyReLU激活
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)  # LeakyReLU激活
        x = x.flatten(start_dim=1)  # 压平
        # x = self.dropout(x)  # Dropout
        x = self.fc(x)  # 通过全连接层
        
        x = torch.sigmoid(x)  # 输出真假概率

        return x