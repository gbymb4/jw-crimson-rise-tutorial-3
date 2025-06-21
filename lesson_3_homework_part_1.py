# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 14:45:00 2025

@author: Gavin
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# Set random seed for reproducibility
torch.manual_seed(42)

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download and load MNIST dataset
print("Loading MNIST dataset...")
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Shared classification head for all models
class ClassificationHead(nn.Module):
    """
    Shared classification head that takes flattened features and outputs class logits
    """
    def __init__(self, input_features, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Flatten the input if it's not already flattened
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)
        return self.classifier(x)


# TODO 1: Implement Basic CNN
class BasicCNN(nn.Module):
    """
    Basic CNN with 3 convolutional layers followed by the classification head
    
    Architecture should be:
    - Conv1: 1->32 channels, 3x3 kernel, padding=1
    - ReLU + MaxPool2d(2x2)
    - Conv2: 32->64 channels, 3x3 kernel, padding=1  
    - ReLU + MaxPool2d(2x2)
    - Conv3: 64->128 channels, 3x3 kernel, padding=1
    - ReLU + MaxPool2d(2x2)
    - Classification head
    
    Input: (batch_size, 1, 28, 28)
    After conv layers: (batch_size, 128, 3, 3) = 1152 features
    """
    
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        self.name = "BasicCNN"
        
        # TODO: Implement the convolutional layers
        # Hint: Use nn.Conv2d, nn.ReLU, nn.MaxPool2d
        # Remember to calculate the correct input size for the classification head
        
        # Your code here:
        self.features = None  # Replace with your convolutional layers
        
        # Pre-implemented classification head
        # After 3 conv+pool layers (28->14->7->3), with 128 channels: 128*3*3 = 1152
        self.classifier = ClassificationHead(1152, num_classes)
    
    def forward(self, x):
        # TODO: Implement the forward pass
        # Apply convolutional layers, then classification head
        
        # Your code here:
        pass


# Helper class for residual connections
class ResidualBlock(nn.Module):
    """
    A residual block with skip connection
    
    Architecture:
    - Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d
    - Add skip connection (input + output)
    - Final ReLU activation
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # TODO: Implement the residual block components
        # Main path: conv -> bn -> relu -> conv -> bn
        # Skip connection: identity or 1x1 conv if channels/size change
        
        # Your code here:
        self.conv1 = None  # Replace with first conv layer
        self.bn1 = None    # Replace with first batch norm
        self.relu = None   # Replace with ReLU activation
        self.conv2 = None  # Replace with second conv layer  
        self.bn2 = None    # Replace with second batch norm
        
        # Skip connection (identity or projection)
        self.skip_connection = None  # Replace with appropriate skip connection
        
    def forward(self, x):
        # TODO: Implement the forward pass
        # Apply main path, add skip connection, apply final activation
        
        # Your code here:
        pass


# TODO 2: Implement Residual CNN
class ResidualCNN(nn.Module):
    """
    CNN with residual connections
    
    Architecture should include:
    - Initial conv layer: 1->64 channels, 3x3 kernel, padding=1
    - 2 Residual blocks (implement ResidualBlock helper class)
    - Final conv layer: 64->128 channels, 3x3 kernel, padding=1
    - Global average pooling or adaptive pooling to reduce spatial dimensions
    - Classification head
    """
    
    def __init__(self, num_classes=10):
        super(ResidualCNN, self).__init__()
        self.name = "ResidualCNN"
        
        # TODO: Implement residual blocks
        # You may want to create a helper ResidualBlock class first
        
        # Your code here:
        self.features = None  # Replace with your layers
        
        # Pre-implemented classification head
        # Use adaptive pooling to get fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # Output: 128*2*2 = 512
        self.classifier = ClassificationHead(512, num_classes)
    
    def forward(self, x):
        # TODO: Implement the forward pass
        # Apply features, adaptive pooling, then classification head
        
        # Your code here:
        pass


# Helper class for multi-scale feature fusion
class MultiScaleBlock(nn.Module):
    """
    Multi-scale convolution block with parallel paths of different kernel sizes
    
    Architecture:
    - Path 1: 3x3 convolution (point-wise)
    - Path 2: 5x5 convolution (standard)  
    - Path 3: 7x7 convolution (large receptive field)
    - Element-wise sum of all paths
    - Final batch normalization and ReLU
    """
    
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        
        # TODO: Implement the three parallel convolution paths
        # All paths should have the same output channels for element-wise addition
        # Use appropriate padding to maintain spatial dimensions
        
        # Your code here:
        self.path1_3x3 = None  # Replace with 3x3 conv: in_channels -> out_channels, padding=1
        self.path2_5x5 = None  # Replace with 5x5 conv: in_channels -> out_channels, padding=2
        self.path3_7x7 = None  # Replace with 7x7 conv: in_channels -> out_channels, padding=3
        
        # Post-fusion processing
        self.bn = None    # Replace with BatchNorm2d(out_channels)
        self.relu = None  # Replace with ReLU activation
        
    def forward(self, x):
        # TODO: Implement the forward pass
        # Apply each path in parallel, sum the outputs, apply bn + relu
        
        # Your code here:
        pass


# TODO 3: Implement Multi-Scale CNN
class MultiScaleCNN(nn.Module):
    """
    CNN with multi-scale feature fusion using parallel convolutions
    
    Architecture should include:
    - Initial conv layer: 1->32 channels, 3x3 kernel, padding=1
    - Multi-scale block with parallel convolutions of different kernel sizes:
      - Path 1: 32->64 channels, 3x3 kernel, padding=1
      - Path 2: 32->64 channels, 5x5 kernel, padding=2
      - Path 3: 32->64 channels, 7x7 kernel, padding=3
    - Element-wise sum of all paths
    - Additional conv layers to reach final feature size
    - Classification head
    """
    
    def __init__(self, num_classes=10):
        super(MultiScaleCNN, self).__init__()
        self.name = "MultiScaleCNN"
        
        # TODO: Implement multi-scale architecture
        # Create parallel convolution paths with different kernel sizes
        # Combine them with element-wise addition
        
        # Your code here:
        self.initial_conv = None  # Replace with initial conv layer
        self.multi_scale_block = None  # Replace with multi-scale block
        self.final_layers = None  # Replace with final conv layers
        
        # Pre-implemented classification head
        # Calculate the correct input size based on your architecture
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Adjust size as needed
        # Assuming final conv outputs 128 channels: 128*4*4 = 2048
        self.classifier = ClassificationHead(2048, num_classes)
    
    def forward(self, x):
        # TODO: Implement the forward pass
        # Apply initial conv, multi-scale block, final layers, then classification head
        
        # Your code here:
        pass

def train_model(model, num_epochs=5):
    """
    Train the model and return training history and final test accuracy
    
    Args:
        model: The CNN model to train
        num_epochs (int): Number of epochs to train for
        
    Returns:
        dict: Training history and final test accuracy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track training history
    train_losses = []
    train_accuracies = []
    
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Print progress every 200 batches
            if batch_idx % 200 == 0:
                print(f'{model.name} - Epoch: {epoch+1}/{num_epochs}, '
                      f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'{model.name} - Epoch {epoch+1}/{num_epochs}, '
              f'Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')
    
    training_time = time.time() - start_time
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_accuracy = correct / total
    print(f'{model.name} - Test Accuracy: {test_accuracy:.4f}')
    print(f'{model.name} - Training Time: {training_time:.1f} seconds\n')
    
    return {
        'model_name': model.name,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracy': test_accuracy,
        'training_time': training_time
    }


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_experiments():
    """
    Run experiments with all three CNN models
    """
    print("Starting CNN Architecture Comparison...")
    
    # TODO: Uncomment these lines once you implement the models
    # models = [
    #     BasicCNN(),
    #     ResidualCNN(), 
    #     MultiScaleCNN()
    # ]
    
    models = []  # Remove this line and uncomment the above once models are implemented
    
    results = []
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"Training {model.name}")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"{'='*50}")
        
        result = train_model(model, num_epochs=5)
        results.append(result)
    
    return results


def plot_results(results):
    """
    Create visualizations comparing the three CNN models
    """
    if not results:
        print("No results to plot. Please implement the models first.")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss
    for result in results:
        ax1.plot(result['train_losses'], label=result['model_name'], linewidth=2)
    ax1.set_title('Training Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy
    for result in results:
        ax2.plot(result['train_accuracies'], label=result['model_name'], linewidth=2)
    ax2.set_title('Training Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test Accuracy Comparison
    model_names = [result['model_name'] for result in results]
    test_accuracies = [result['test_accuracy'] for result in results]
    
    bars = ax3.bar(model_names, test_accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax3.set_title('Final Test Accuracy Comparison')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Training Time Comparison
    training_times = [result['training_time'] for result in results]
    
    bars = ax4.bar(model_names, training_times, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax4.set_title('Training Time Comparison')
    ax4.set_ylabel('Training Time (seconds)')
    
    # Add value labels on bars
    for bar, timestamp in zip(bars, training_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{timestamp:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nEXPERIMENT SUMMARY:")
    print("="*60)
    for result in results:
        print(f"{result['model_name']:15} - "
              f"Test Accuracy: {result['test_accuracy']:.4f}, "
              f"Training Time: {result['training_time']:.1f}s, "
              f"Parameters: {count_parameters(globals()[result['model_name']]().cuda() if torch.cuda.is_available() else globals()[result['model_name']]())}")


# Main execution
if __name__ == "__main__":
    # Run experiments
    results = run_experiments()
    
    # Plot results
    plot_results(results)
    
    print("\n" + "="*60)
    print("HOMEWORK ANALYSIS QUESTIONS:")
    print("="*60)
    print("1. Which CNN architecture achieved the highest test accuracy?")
    print("2. How do the parameter counts compare between models?")
    print("3. Which model trained fastest? Why might this be?")
    print("4. What are the advantages of residual connections?")
    print("5. How does multi-scale feature fusion help with classification?")
    print("6. What trade-offs do you observe between accuracy and training time?")
    print("\nAdd your answers to homework_questions.md")
    print("="*60)