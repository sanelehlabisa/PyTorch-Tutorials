# ==================================
#       Creating Castom Dataset     
# ==================================
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class CatAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)        

        return (image, y_label)





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
batch_size = 64
num_epochs = 128

# Load datasets
dataset = CatAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized', transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [8, 2])

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Initialize model which is from pytorch
model = torchvision.models.googlenet(pretrained=True).to(device)

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
