# =====================================================
#           Data Augmentation using Torchvision        
# =====================================================

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image
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
    

def main():
    my_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomRotation(degrees=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ]
    )
    dataset = CatAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized', transform=my_transforms)

    img_num = 0
    for _ in range(5):
        for img, lable in dataset:
            save_image(img, f'img{img_num}.png')
            img_num += 1

if __name__ == '__main__':
    main()