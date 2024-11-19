# ==============================================================
#       Creating, Training and Evaluating a Convolutional Neural Network      
# ==============================================================

# Imports
import torch
import torch.nn as nn # Contains neural network functions
import torch.optim as optim # Contains optimizers
import torch.nn.functional as F # Containers activation functions
from torch.utils.data import DataLoader # Contains functions for easy dataset management
import torchvision.datasets as dataset # Contains a lot of simple datasets
import torchvision.transforms as transforms # Contains transofrmations that can be performed in a dataset

# Creating Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc_1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_1(x)
        return x

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 32
num_epochs = 128
load_model = True


# Load Data
train_dataset = dataset.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataset.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
#model = NN(input_size, num_classes).to(device)
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load Model
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

# Train the model
for epoch in range(num_epochs):
    loss = 0
    if epoch % 5:
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        target = target.to(device=device)
        
        # Forward
        scores = model(data)
        loss += criterion(scores, target)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update gradient based of the loss
    optimizer.step()

    # Print epoch and loss
    print(f"Epoch: {epoch} loss: {loss}")

# Check accuracy on test data
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, prediction =  scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)
        print(f"Accuracy: {(num_correct / num_samples) * 100:.2f}")

check_accuracy(test_loader, model)