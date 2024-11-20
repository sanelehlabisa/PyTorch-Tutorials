# =========================================================================
#       Creating, Training and Evaluating a Pretrained VGG Model
# =========================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 5

# Transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),  # Convert to Tensor
])

# Load datasets
train_dataset = dataset.MNIST(root='dataset/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataset.MNIST(root='dataset/', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Define the model
class VGG16Modified(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16Modified, self).__init__()
        # Load pretrained VGG16
        self.vgg = torchvision.models.vgg16(pretrained=True)
        
        # Freeze feature extractor layers
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 128),  # Adjust input size to 25088
            nn.ReLU(),  # Non-linearity
            nn.Dropout(0.5),  # Regularization
            nn.Linear(128, num_classes)  # Final output layer
        )
    
    def forward(self, x):
        x = self.vgg(x)  # Forward through VGG16
        return x

# Initialize model
model = VGG16Modified(num_classes=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Send data to device
        data = data.to(device)
        target = target.to(device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Evaluation function
def check_accuracy(loader, model):
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f"Accuracy: {float(num_correct) / num_samples * 100:.2f}%")

# Check accuracy
check_accuracy(test_loader, model)
