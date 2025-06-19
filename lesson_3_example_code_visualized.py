# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 19:57:39 2025

@author: Gavin
"""

# -*- coding: utf-8 -*-  
"""Created on Thu Jun 19 18:50:03 2025  
@author: Gavin"""  
  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
import torchvision  
import torchvision.transforms as transforms  
import matplotlib.pyplot as plt  
import seaborn as sns  
import numpy as np  
  
# Set random seed for reproducibility  
torch.manual_seed(42)  
  
# Determine the device  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# CIFAR-10 class names  
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',  
               'dog', 'frog', 'horse', 'ship', 'truck']  
  
# Data preprocessing  
transform = transforms.Compose([  
    transforms.ToTensor(),  # Already scales images to [0, 1]  
])  
  
# Load CIFAR-10 dataset  
print("Loading CIFAR-10 dataset...")  
train_dataset = torchvision.datasets.CIFAR10(  
    root='./data', train=True, download=True, transform=transform)  
test_dataset = torchvision.datasets.CIFAR10(  
    root='./data', train=False, download=True, transform=transform)  
  
# Create data loaders  
batch_size = 512  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
  
# Show sample images  
def show_sample_images():  
    dataiter = iter(train_loader)  
    images, labels = next(dataiter)  
  
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))  
    for i in range(8):  
        row = i // 4  
        col = i % 4  
  
        img = images[i].permute(1, 2, 0)  # Change from (C,H,W) to (H,W,C)  
  
        axes[row, col].imshow(img)  
        axes[row, col].set_title(f'{class_names[labels[i].item()]}')  
        axes[row, col].axis('off')  
    plt.suptitle('Sample CIFAR-10 Images')  
    plt.tight_layout()  
    plt.show()  
  
show_sample_images()  
  
# Define CNN Architecture  
class CIFAR10CNN(nn.Module):  
    def __init__(self):  
        super(CIFAR10CNN, self).__init__()  
          
        # First convolutional block  
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32x32x32  
        self.relu1 = nn.ReLU()  
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)       # 16x16x32  
          
        # Second convolutional block  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 16x16x64  
        self.relu2 = nn.ReLU()  
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)       # 8x8x64  
          
        # Third convolutional block  
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # 8x8x64  
        self.relu3 = nn.ReLU()  
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)       # 4x4x64  
          
        # Flatten layer  
        self.flatten = nn.Flatten()  
          
        # Fully connected layers  
        self.fc1 = nn.Linear(4 * 4 * 64, 128)  # 4x4x64 = 1024  
        self.relu4 = nn.ReLU()  
        self.dropout = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(128, 10)  # 10 classes  
  
    def forward(self, x):  
        # First conv block  
        x = self.conv1(x)  
        x = self.relu1(x)  
        x = self.pool1(x)  
          
        # Second conv block  
        x = self.conv2(x)  
        x = self.relu2(x)  
        x = self.pool2(x)  
          
        # Third conv block  
        x = self.conv3(x)  
        x = self.relu3(x)  
        x = self.pool3(x)  
          
        # Flatten and fully connected  
        x = self.flatten(x)  
        x = self.fc1(x)  
        x = self.relu4(x)  
        x = self.dropout(x)  
        x = self.fc2(x)  
          
        return x  
  
# Create model, loss function, and optimizer  
model = CIFAR10CNN().to(device)  # Move model to the device  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.005)  
  
print("Model Architecture:")  
print(model)  
  
# Count parameters  
total_params = sum(p.numel() for p in model.parameters())  
print(f"\nTotal parameters: {total_params:,}")  
  
# Training function  
def train_model(num_epochs=10):  
    model.train()  
    train_losses = []  
    train_accuracies = []  
  
    for epoch in range(num_epochs):  
        total_loss = 0  
        correct = 0  
        total = 0  
  
        for batch_idx, (data, targets) in enumerate(train_loader):  
            data, targets = data.to(device), targets.to(device)  # Move data to the device  
  
            # Forward pass  
            outputs = model(data)  
            loss = criterion(outputs, targets)  
  
            # Backward pass  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
  
            # Statistics  
            total_loss += loss.item()  
            _, predicted = torch.max(outputs.data, 1)  
            total += targets.size(0)  
            correct += (predicted == targets).sum().item()  
  
            # Print progress  
            if batch_idx % 200 == 0:  
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, '  
                      f'Loss: {loss.item():.4f}')  
  
        # Calculate epoch statistics  
        epoch_loss = total_loss / len(train_loader)  
        epoch_acc = correct / total  
        train_losses.append(epoch_loss)  
        train_accuracies.append(epoch_acc)  
  
        print(f'Epoch {epoch+1}/{num_epochs}: '  
              f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')  
    return train_losses, train_accuracies  
  
# Testing function  
def test_model():  
    model.eval()  
    correct = 0  
    total = 0  
    class_correct = list(0. for i in range(10))  
    class_total = list(0. for i in range(10))  
  
    with torch.no_grad():  
        for data, targets in test_loader:  
            data, targets = data.to(device), targets.to(device)  # Move data to the device  
  
            outputs = model(data)  
            _, predicted = torch.max(outputs, 1)  
            total += targets.size(0)  
            correct += (predicted == targets).sum().item()  
  
            # Per-class accuracy  
            c = (predicted == targets).squeeze()  
            for i in range(targets.size(0)):  
                label = targets[i]  
                class_correct[label] += c[i].item()  
                class_total[label] += 1  
  
    # Overall accuracy  
    accuracy = correct / total  
    print(f'\nTest Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)')  
  
    # Per-class accuracy  
    print("\nPer-class accuracy:")  
    for i in range(10):  
        if class_total[i] > 0:  
            class_acc = class_correct[i] / class_total[i]  
            print(f'{class_names[i]}: {class_acc:.4f} ({100 * class_acc:.1f}%)')  
    return accuracy  
  
# Visualization function  
def show_predictions():  
    model.eval()  
    dataiter = iter(test_loader)  
    images, labels = next(dataiter)  
    images, labels = images.to(device), labels.to(device)  # Move data to the device  
  
    with torch.no_grad():  
        outputs = model(images)  
        _, predicted = torch.max(outputs, 1)  
  
        # Show first 8 predictions  
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))  
        for i in range(8):  
            row = i // 4  
            col = i % 4  
  
            img = images[i].cpu().permute(1, 2, 0)  
  
            axes[row, col].imshow(img)  
            true_label = class_names[labels[i].item()]  
            pred_label = class_names[predicted[i].item()]  
            axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}')  
            axes[row, col].axis('off')  
  
            # Color based on correctness  
            if labels[i].item() == predicted[i].item():  
                axes[row, col].title.set_color('green')  
            else:  
                axes[row, col].title.set_color('red')  
        plt.suptitle('CNN Predictions on CIFAR-10')  
        plt.tight_layout()  
        plt.show()  
  
# Train the model  
print("\nStarting training...")  
train_losses, train_accuracies = train_model(num_epochs=50)  
  
# Test the model  
print("\nTesting the model...")  
test_accuracy = test_model()  
  
# Show training progress  
plt.figure(figsize=(12, 4))  
plt.subplot(1, 2, 1)  
plt.plot(train_losses, 'b-', label='Training Loss')  
plt.title('Training Loss Over Epochs')  
plt.xlabel('Epoch')  
plt.ylabel('Loss')  
plt.legend()  
plt.grid(True)  
  
plt.subplot(1, 2, 2)  
plt.plot([acc * 100 for acc in train_accuracies], 'r-', label='Training Accuracy')  
plt.title('Training Accuracy Over Epochs')  
plt.xlabel('Epoch')  
plt.ylabel('Accuracy (%)')  
plt.legend()  
plt.grid(True)  
plt.tight_layout()  
plt.show()  
  
# Show predictions  
show_predictions()  
  
# Visualize learned filters  
def visualize_filters():  
    # Get first convolutional layer weights  
    first_layer = model.conv1  
    filters = first_layer.weight.data  
  
    # Plot first 16 filters  
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))  
    for i in range(16):  
        row = i // 4  
        col = i % 4  
  
        # Normalize filter for display  
        filter_img = filters[i].permute(1, 2, 0)  # Change to (H,W,C)  
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())  
  
        axes[row, col].imshow(filter_img.cpu())  
        axes[row, col].set_title(f'Filter {i+1}')  
        axes[row, col].axis('off')  
    plt.suptitle('Learned Convolutional Filters (First Layer)')  
    plt.tight_layout()  
    plt.show()  
  
visualize_filters()  
  
# Visualize the feature map  
def visualize_feature_maps():  
    model.eval()  
    dataiter = iter(test_loader)  
    images, _ = next(dataiter)  
    images = images.to(device)  
  
    with torch.no_grad():  
        # Forward pass up to the first conv layer  
        x = model.conv1(images[0].unsqueeze(0))  
        x = model.relu1(x)  
  
    # Convert feature maps to numpy for plotting  
    feature_maps = x.cpu().numpy().squeeze()  
  
    # Plot the feature maps  
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))  
    for i in range(32):  # 32 feature maps in the first conv layer  
        row = i // 8  
        col = i % 8  
        axes[row, col].imshow(feature_maps[i], cmap='viridis')  
        axes[row, col].axis('off')  
    plt.suptitle('Feature Maps from First Convolutional Layer')  
    plt.tight_layout()  
    plt.show()  
  
visualize_feature_maps()  
  
# Plot a confusion matrix  
def plot_confusion_matrix():  
    from sklearn.metrics import confusion_matrix  
    import seaborn as sns  
  
    model.eval()  
    all_preds = torch.tensor([]).to(device)  
    all_labels = torch.tensor([]).to(device)  
  
    with torch.no_grad():  
        for data, targets in test_loader:  
            data, targets = data.to(device), targets.to(device)  
            outputs = model(data)  
            _, preds = torch.max(outputs, 1)  
            all_preds = torch.cat((all_preds, preds))  
            all_labels = torch.cat((all_labels, targets))  
  
    cm = confusion_matrix(all_labels.cpu(), all_preds.cpu())  
    plt.figure(figsize=(10, 8))  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)  
    plt.xlabel('Predicted')  
    plt.ylabel('True')  
    plt.title('Confusion Matrix')  
    plt.show()  
  
plot_confusion_matrix()  
