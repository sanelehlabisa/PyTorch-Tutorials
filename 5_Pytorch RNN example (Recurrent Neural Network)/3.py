# ==============================================================================================
#       Creating, Training and Evaluating a Long Short Term Memory Recurrent Neural Network      
# ==============================================================================================

# Imports
import torch
import torch.nn as nn # Contains neural network functions
import torch.optim as optim # Contains optimizers
import torch.nn.functional as F # Containers activation functions
from torch.utils.data import DataLoader # Contains functions for easy dataset management
import torchvision.datasets as dataset # Contains a lot of simple datasets
import torchvision.transforms as transforms # Contains transofrmations that can be performed in a dataset

# Creating Long Short Term Memory Recurrent Neural Network
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 10
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 4

# Load Data
train_dataset = dataset.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = dataset.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, prediction =  scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)
        print(f"Accuracy: {(num_correct / num_samples) * 100:.2f}")

check_accuracy(test_loader, model)