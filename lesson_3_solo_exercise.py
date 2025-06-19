# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 20:02:56 2025

@author: Gavin
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Advanced data augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

# Load data with augmentation
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class ConvBlock(nn.Module):
    """A convolutional block with Conv2d + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """A simplified residual block with skip connection"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class AdvancedCIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AdvancedCIFAR10CNN, self).__init__()
        
        # TODO: Design your advanced architecture
        # Guidelines:
        # 1. Start with a conv block to process input
        # 2. Add multiple stages with increasing channels
        # 3. Use residual blocks for deeper learning
        # 4. Use global average pooling instead of fully connected layers
        # 5. Add dropout for regularization
        
        # Stage 1: Initial feature extraction
        # TODO: Add initial convolutional layers
        
        # Stage 2-4: Progressive feature extraction with residual connections
        # TODO: Add multiple stages with ResidualBlocks
        
        # Global Average Pooling + Classifier
        # TODO: Add global average pooling and final classifier
        
        pass
    
    def forward(self, x):
        # TODO: Implement the forward pass
        # Process through all stages
        # Apply global average pooling
        # Return classification logits
        
        pass

# Training with advanced techniques
def train_advanced_model(model, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}')
        
        scheduler.step()
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{num_epochs}: Accuracy: {epoch_acc:.4f}')

# TODO: Complete the implementation and train your model
# model = AdvancedCIFAR10CNN()
# train_advanced_model(model, num_epochs=20)