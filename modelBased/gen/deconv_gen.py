import torch.nn as nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import math


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
        self.z_size = z_shape
        # ConvTranspose: Output size= (Input_size − 1) × Stride − 2 * Padding + Kernel_size
        layer1 = {
            'in_channels': z_shape,
            'out_channels': 256,
            'kernel_size': 3,
            'stride': 1,
            'padding': 0,
            'bias': False
        }
        
        output_size1 =  (1 - 1) * layer1['stride'] - 2 * layer1['padding'] + layer1['kernel_size']

        layer2 = {
            'in_channels': layer1['out_channels'],
            'out_channels': 128,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'bias': False
        }
        
        output_size2 =  (output_size1 - 1) * layer2['stride'] - 2 * layer2['padding'] + layer2['kernel_size']
        kernel_size_w, stride_w, padding_w = compute_deconv_params(output_size2, width)
        kernel_size_h, stride_h, padding_h = compute_deconv_params(output_size2, height)

        layer3 = {
            'in_channels': layer2['out_channels'],
            'out_channels': num_classes,
            'kernel_size': (kernel_size_h, kernel_size_w),
            'stride': (stride_h, stride_w),
            'padding': (padding_h, padding_w),
            'bias': False
        }

        self.main = nn.Sequential(
            # First layer: From (1,1) to (3,3)
            nn.ConvTranspose2d(**layer1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Second layer: From (3,3) to (6,6)
            nn.ConvTranspose2d(**layer2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Third layer: From (6,6) to (width, height)
            nn.ConvTranspose2d(**layer3)
        )

    def forward(self, z):
        x = z.reshape(-1, self.z_size, 1, 1)
        x = self.main(x)
        x = nn.Softmax(dim=1)(x)  # Apply softmax over the class dimension
        return x  # Output shape: (batch_size, 6, 6, 12)

class Discriminator(nn.Module):
    def __init__(self, input_channels, width, height):
        super(Discriminator, self).__init__()
        ndf = 64  
        self.width = width
        self.height = height
        self.num_classes = input_channels

        # Conv: Output size= (input_size + 2 * Padding - Kernel_size) / Stride + 1
        layer1 = {
            'in_channels':input_channels,
            'out_channels': ndf,
            'kernel_size': 3,
            'stride': 2,
            'padding': 1,
            'bias': False
        }
        output_size1 = (width + 2 * layer1['padding'] - layer1['kernel_size']) // layer1['stride'] + 1

        layer2 = {
            'in_channels': layer1['out_channels'],
            'out_channels': 2 * layer1['out_channels'],
            'kernel_size': 3,
            'stride': 2,
            'padding': 1,
            'bias': False
        }
        output_size2 = (output_size1 + 2 * layer2['padding'] - layer2['kernel_size']) // layer2['stride'] + 1

        layer3 = {
            'in_channels': layer2['out_channels'],
            'out_channels': 1,
            'kernel_size': 2,
            'stride': 1,
            'padding': 0,
            'bias': False
        }
        output_layer3 = (output_size2 + 2 * layer3['padding'] - layer3['kernel_size']) // layer3['stride'] + 1


        self.main = nn.Sequential(
            nn.Conv2d(**layer1),  
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(dropout),

            # Second layer: Adjusted kernel size to (2, 4)
            nn.Conv2d(**layer2),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(dropout),

            nn.Conv2d(**layer3 ),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.shape[1:] == (self.height, self.width):
            reshaped_input = input.view(-1)  

            # Apply one-hot encoding
            one_hot_encoded = F.one_hot(reshaped_input, self.num_classes)  
            input = one_hot_encoded.view(input.shape[0], self.height, self.width, self.num_classes).permute(0, 3, 1, 2)  # Shape (32, 6, 8, 8)

        input = input.float()  
        output = self.main(input)  
        return output.view(-1)  