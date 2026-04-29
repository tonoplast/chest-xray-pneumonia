
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# simple CNN
### image -> conv blocks -> flatten -> MLP -> output
class ConvBlock(nn.Module):
    ''' single conv block: conv -> batchnorm -> relu -> maxpool '''
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # 3x3 filter, padding to keep spatial size same
            nn.BatchNorm2d(out_channels), # help trianing stability, speed (stabilises gradients, reduces internal covariate shift)
            nn.ReLU(inplace=True), # non-linearity, saves memory
            nn.MaxPool2d(kernel_size=2, stride=stride), # downsample by 2, halves width and height (reduce computation, adds translation invariance)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    

class PneumoniaCNN(nn.Module):
    ''' simple cnn for binary peumonia classification '''
    
    def __init__(self, image_size: int = 64) -> None:
        super().__init__()
        
        self.features = nn.Sequential( # [B,   1, 64, 64]
            ConvBlock(1, 32),          # [B,  32, 32, 32]
            ConvBlock(32, 64),         # [B,  64, 16, 16]
            ConvBlock(64, 128),        # [B, 128,  8,  8]
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # calculate flattened size after convolutions
        # reduced = image_size // (2 ** 3) # 3 pooling layer, each halves size (64 / 8 = 8)
        # flat_size = 128 * reduced * reduced # 128 * 8 * 8 = 8192, need adaptive pooling if this changes
        flat_size = 128
        
        self.classifier = nn.Sequential(
            nn.Flatten(),               # [B, 128, 8, 8] -> [B, 8192]
            nn.Linear(flat_size, 256),  # [8192 -> 256] (if no AdaptiveAvgPool2d, else [128 -> 256])
            nn.ReLU(inplace=True),      #
            nn.Dropout(p=0.5),          # prevent overfitting (Dropout 50%)
            nn.Linear(256, 1)           # 256 -> 1 (single number/logit)
        )
        
        logger.info(f'model initialised with image_size={image_size}, flat_size={flat_size}')
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# residual style
class ResidualBlock(nn.Module):
    ''' simple ResNet-style block: conv -> bn -> relu -> conv -> bn + skip connection '''
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            
        )
    
        #If shape changes, make skip path math main path
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main(x) + self.skip(x))


class PneumoniaResNetCNN(nn.Module):
    ''' small ResNet-style CNN for binary pneumonia classification '''
    
    def __init__(self) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False), # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.features = nn.Sequential(
            ResidualBlock(32, 32, stride=1),   #[B, 32, 64, 64]
            ResidualBlock(32, 64, stride=2),   #[B, 64, 32, 32]
            ResidualBlock(64, 128, stride=2),  #[B, 128, 16, 16]
            ResidualBlock(128, 128, stride=2), #[B, 128, 8, 8]
            
            nn.AdaptiveAvgPool2d((1, 1)),      #[B, 128, 1, 1]
        )
        
        flat_size = 128
        
        self.classifier = nn.Sequential(
            nn.Flatten(),           #[B, 128]
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),   # single logit
        )
        
        logger.info("ResNet-style model initialised")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)