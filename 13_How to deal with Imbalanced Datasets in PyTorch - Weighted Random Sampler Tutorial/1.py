# ==========================================
#       Dealing with Imbalance datasets    
# ==========================================

import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

# Methods fpr daeling with imbalance datasets
#   1. Oversampling
#   2. Class weighting

def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    class_weights: list[float] = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))
    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampleer = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampleer)
    return loader

def main():
    loader = get_loader(root_dir="dogs_dataset", batch_size=8)

    for data, label in loader:
        print(label)

if __name__ == "__main__":
    main()
