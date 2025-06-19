# Deep Learning with PyTorch - Session 3

## Session Timeline

| Time      | Activity                                |
| --------- | --------------------------------------- |
| 0:00 - 0:10 | 1. Check-in + Session 2 Recap          |
| 0:10 - 0:25 | 2. CNN Theory and Motivation           |
| 0:25 - 0:45 | 3. Complete CIFAR-10 CNN Example       |
| 0:45 - 1:10 | 4. Code Walkthrough and Discussion     |
| 1:10 - 1:45 | 5. Advanced CNN Architecture Challenge |
| 1:45 - 2:00 | 6. Wrap-Up + Next Steps                |

---

## 1. Check-in + Session 2 Recap

### Goals

Reconnect and review what we learned about image classification before diving into convolutional neural networks!

### Quick Recap Questions

* How did your MNIST neural network perform? What accuracy did you achieve?
* What was the difference between classification and regression?
* Can you explain what ReLU activation functions do?
* How did CrossEntropyLoss work compared to MSELoss?
* Did anyone try the homework challenges with Fashion-MNIST or hyperparameter tuning?

### Session 2 Key Concepts

* **Multi-layer neural networks** with fully connected layers
* **Image classification** using flattened pixel vectors
* **Training/testing loops** with accuracy metrics
* **MNIST dataset** - 28x28 grayscale handwritten digits

### What's New Today

Today we're taking a big step forward to **Convolutional Neural Networks (CNNs)**! We'll work with **CIFAR-10**, which contains 32x32 **color images** of real-world objects like airplanes, cars, and animals.

This is much more challenging than MNIST because:
- **Color images** (3 channels: RGB) instead of grayscale
- **Real-world objects** instead of simple digits
- **More complex patterns** that require spatial understanding

---

## 2. CNN Theory and Motivation

### Goals

* Understand why fully connected networks struggle with images
* Learn how convolutional layers work
* Build intuition for CNN architectures
* Understand the key components: convolution, pooling, filters

---

### Why Do We Need CNNs?

**Problem with Fully Connected Networks:**
- MNIST: 28×28×1 = 784 parameters for first layer
- CIFAR-10: 32×32×3 = 3,072 parameters for first layer
- High-resolution image (224×224×3): 150,528 parameters just for the first layer!

**More importantly:** Fully connected layers don't understand spatial relationships. A cat's ear should be recognized whether it's in the top-left or top-right of the image.

### How Convolution Works

**Key Insight:** Instead of looking at the entire image at once, look at small patches and learn to detect patterns like edges, textures, and shapes.

**Convolution Operation:**
- Take a small **filter** (like 3×3 pixels)
- Slide it across the image
- At each position, compute a dot product
- This creates a **feature map** showing where the pattern was found

**Example Filters:**
- Edge detectors (horizontal/vertical lines)
- Texture detectors (rough/smooth surfaces)  
- Shape detectors (curves, corners)

### CNN Architecture Components

1. **Convolutional Layers** (`nn.Conv2d`)
   - Apply filters to detect features
   - Parameters: input channels, output channels, kernel size
   - Example: `nn.Conv2d(3, 32, kernel_size=3)`

2. **Activation Functions** (`nn.ReLU`)
   - Add non-linearity (same as before)
   - Applied after each convolution

3. **Pooling Layers** (`nn.MaxPool2d`)
   - Reduce spatial dimensions
   - Make the network less sensitive to exact positions
   - Example: `nn.MaxPool2d(kernel_size=2, stride=2)`

4. **Fully Connected Layers** (`nn.Linear`)
   - At the end, for final classification
   - Same as our MNIST network

### Typical CNN Architecture Flow

```
Input Image (32×32×3)
    ↓
Conv2d(3→32) + ReLU → Feature maps (32×32×32)
    ↓
MaxPool2d → Smaller feature maps (16×16×32)
    ↓
Conv2d(32→64) + ReLU → More feature maps (16×16×64)
    ↓
MaxPool2d → Even smaller (8×8×64)
    ↓
Flatten → Vector (8×8×64 = 4,096)
    ↓
Linear layers → Classification (10 classes)
```

**Key Benefits:**
- **Parameter sharing:** Same filter used across the entire image
- **Translation invariance:** Recognizes patterns regardless of position
- **Hierarchical features:** Early layers detect edges, later layers detect objects

---

## 3. Complete CIFAR-10 CNN Example

### Goals

* See a working CNN that classifies color images
* Understand the complete pipeline from data loading to evaluation
* Observe the performance difference between CNNs and fully connected networks

---

### The CIFAR-10 Dataset

**What is CIFAR-10?**
- 60,000 color images (32×32 pixels)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images, 10,000 test images
- Much more challenging than MNIST!

---

### Complete CIFAR-10 CNN Implementation

```python
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
import torchvision  
import torchvision.transforms as transforms  
import matplotlib.pyplot as plt  
  
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
  
        # No need to denormalize, images are already in [0, 1]  
        img = images[i].permute(1, 2, 0)   # Change from (C,H,W) to (H,W,C)  
  
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
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)        # 4x4x64 
  
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
optimizer = optim.Adam(model.parameters(), lr=0.001)  
  
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
  
            # No need to denormalize, images are already in [0, 1]  
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
train_losses, train_accuracies = train_model(num_epochs=25)  
  
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
```

### Expected Results

With this CNN architecture, you should achieve:
- **Training accuracy**: 85-90% after 10 epochs
- **Test accuracy**: 70-80% (significantly better than a fully connected network)
- **Visual filters**: The first layer learns edge and color detectors

---

## 4. Code Walkthrough and Discussion

### Goals

* Understand each component of the CNN architecture
* Learn how to calculate layer dimensions
* Understand the role of normalization and dropout
* Compare CNN vs. fully connected network performance

---

### Architecture Deep Dive

#### 1. **Data Preprocessing**
```python
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```
- Normalizes each RGB channel from [0,1] to [-1,1]
- Helps with training stability and convergence
- The values (0.5, 0.5, 0.5) are the mean and std for each channel

#### 2. **Convolutional Layers**
```python
nn.Conv2d(3, 32, kernel_size=3, padding=1)
```
- **3 input channels** (RGB)
- **32 output channels** (32 different filters)
- **3×3 kernel size** (small filters to detect local patterns)
- **padding=1** keeps the spatial dimensions the same

#### 3. **Pooling Layers**
```python
nn.MaxPool2d(kernel_size=2, stride=2)
```
- Reduces spatial dimensions by half
- Makes the network more robust to small translations
- Reduces computational requirements

#### 4. **Dimension Tracking**
```
Input: 32×32×3
Conv1 + MaxPool1: 16×16×32
Conv2 + MaxPool2: 8×8×64  
Conv3 + MaxPool3: 4×4×128
Flatten: 4×4×128 = 2,048
```

#### 5. **Dropout Regularization**
```python
nn.Dropout(0.5)
```
- Randomly sets 50% of neurons to zero during training
- Prevents overfitting
- Only active during training, turned off during testing

### Discussion Questions

* **Architecture Choices**: Why do we increase the number of filters (32→64→128) as we go deeper?
* **Parameter Efficiency**: How many parameters does this CNN have compared to a fully connected network?
* **Feature Hierarchy**: What kinds of features do you think each layer learns?
* **Performance**: Why does the CNN perform better than fully connected networks on images?

### Key Insights

* **Early layers** learn low-level features (edges, textures)
* **Later layers** learn high-level features (shapes, objects)
* **Spatial structure** is preserved through convolutional operations
* **Parameter sharing** makes CNNs much more efficient than fully connected networks

---

## 5. Advanced CNN Architecture Challenge

### Goal

Design and implement a more sophisticated CNN architecture using modern techniques like batch normalization, deeper networks, and advanced pooling strategies.

---

### Challenge: Build a ResNet-Inspired CNN

Your task is to create a more advanced CNN that incorporates:
1. **Batch normalization** for better training stability
2. **Deeper architecture** with more convolutional layers
3. **Skip connections** (simplified ResNet concept)
4. **Global average pooling** instead of fully connected layers
5. **Data augmentation** for better generalization

### Advanced CNN Template

```python
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
```

### Challenge Requirements

1. **Complete the AdvancedCIFAR10CNN class**:
   - Use ConvBlock and ResidualBlock building blocks
   - Create 4 stages with increasing filter counts (32→64→128→256)
   - Add skip connections via ResidualBlocks
   - Use global average pooling before classification

2. **Key Components to Include**:
   - Batch normalization in all convolutional layers
   - At least 2 residual blocks per stage
   - Global average pooling instead of flatten + linear
   - Dropout before the final classifier

3. **Expected Architecture Flow**:
   ```
   Input (3×32×32)
   ↓ Initial ConvBlock
   Stage 1: ConvBlock + ResidualBlocks (32 filters)
   ↓ Downsample
   Stage 2: ConvBlock + ResidualBlocks (64 filters)  
   ↓ Downsample
   Stage 3: ConvBlock + ResidualBlocks (128 filters)
   ↓ Downsample  
   Stage 4: ConvBlock + ResidualBlocks (256 filters)
   ↓ Global Average Pooling
   Classifier → 10 classes
   ```

### Advanced Concepts to Implement

**Batch Normalization**: Normalizes inputs to each layer, leading to faster and more stable training.

**Residual Connections**: Allow gradients to flow directly through skip connections, enabling much deeper networks.

**Global Average Pooling**: Replaces fully connected layers by averaging each feature map to a single value.

**Data Augmentation**: Creates variations of training images to improve generalization.

### Target Performance

With a well-designed advanced CNN, you should achieve:
- **85-90% test accuracy** (10-15% improvement over basic CNN)
- **Faster convergence** due to batch normalization
- **Better generalization** due to data augmentation and regularization

---

## 6. Wrap-Up + Next Steps

### Session 3 Recap

Today we learned:

* **Convolutional Neural Networks** and why they're better for images
* **CIFAR-10 dataset** with color images and real-world objects
* **CNN components**: convolution, pooling, filters, feature maps  
* **Complete CNN implementation** from data loading to evaluation
* **Advanced techniques**: batch normalization, residual connections, data augmentation

### Key Concepts Mastered

* **Spatial feature extraction** through convolutional layers
* **Hierarchical learning** from edges to objects
* **Parameter efficiency** through weight sharing
* **Translation invariance** through convolution and pooling
* **Modern CNN techniques** for better performance

### Performance Comparison

* **Fully connected on MNIST**: 95-98% accuracy
* **CNN on MNIST**: 99%+ accuracy
* **Fully connected on CIFAR-10**: 45-55% accuracy
* **Basic CNN on CIFAR-10**: 70-80% accuracy
* **Advanced CNN on CIFAR-10**: 85-90% accuracy

### Homework Projects

Choose one or more projects to deepen your CNN understanding:

#### 1. **Complete the Advanced CNN Challenge**
* Finish implementing the ResNet-inspired architecture
* Train for 20+ epochs and achieve 85%+ accuracy
* Compare with the basic CNN performance
* Experiment with different numbers of residual blocks

#### 2. **CNN Architecture Exploration**
* Try different kernel sizes (1×1, 5×5, 7×7)
* Experiment with different pooling strategies (average pooling, adaptive pooling)
* Test the effect of different depths (shallow vs. deep networks)
* Create a report comparing different architectural choices

#### 3. **Transfer Learning with Pretrained Models**
* Load a pretrained ResNet or VGG model
* Fine-tune it on CIFAR-10
* Compare with your CNN trained from scratch
* Observe how much faster it converges

```python
import torchvision.models as models

# Load pretrained ResNet
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # Adapt for CIFAR-10
```

#### 4. **Custom Dataset Challenge**
* Create your own image classification dataset
* Use your smartphone to take photos of 3-5 different objects/categories
* Resize images to 32×32 and create a custom dataset
* Train your CNN and see how it performs on your personal data

#### 5. **Visualization and Interpretation**
* Implement Grad-CAM to visualize which parts of images the CNN focuses on
* Create activation maps for different layers
* Analyze misclassified examples and understand failure modes
* Plot training curves and analyze overfitting patterns

### Next Session Preview

In Session 4, we'll explore:
* **Transfer Learning** with pretrained models
* **Advanced datasets** (ImageNet, custom datasets)
* **Object detection** and segmentation
* **Generative models** and autoencoders
* **Deployment** of trained models

### Advanced Resources

**Papers to Read:**
* LeNet-5: Original CNN paper
* AlexNet: Deep CNN breakthrough
* ResNet: Skip connections and very deep networks
* VGG: Systematic architecture design

**Datasets to Explore:**
* **CIFAR-100**: 100 classes instead of 10
* **STL-10**: Higher resolution images
* **ImageNet**: 1000 classes, millions of images
* **Custom datasets**: Create your own!

**Libraries and Tools:**
* **TorchVision models**: Pretrained architectures
* **Albumentations**: Advanced data augmentation
* **TensorBoard**: Training visualization
* **Gradio/Streamlit**: Model deployment

### Technical Setup

Make sure you have these packages for the homework:

```bash
pip install torch torchvision torchaudio
pip install matplotlib seaborn
pip install scikit-learn
pip install albumentations  # For advanced data augmentation
pip install grad-cam        # For visualization
```

**GPU Acceleration:**
If you have a CUDA-compatible GPU:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')