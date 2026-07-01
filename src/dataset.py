import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from utils import extract_features
import os
import numpy as np

class LensDataset(Dataset):
    def __init__(self, csv_file, base_img_dir, transform=None,
                  augment=False):
        self.data_frame   = pd.read_csv(csv_file)
        self.base_img_dir = base_img_dir
        self.augment      = augment

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        row    = self.data_frame.iloc[index]
        target = np.fromstring(row["confidence_vector"].strip("[]"),sep=" ")

        target = torch.tensor(target, dtype=torch.float32)

        img_path = row['ae_image_path']
        image    = Image.open(img_path).convert('RGB')

        if self.transform:
            img_tensor = self.transform(image)
        elif self.augment:
            img_tensor = self.train_transform(image)
        else:
            img_tensor = self.val_transform(image)

        
        feature_vector = extract_features(img_tensor)

        return feature_vector, target


def get_dataloader(train_csv, val_csv, base_img_dir, batch_size=64):
    train_dataset = LensDataset(train_csv, base_img_dir,
                                augment=True)
    val_dataset   = LensDataset(val_csv,   base_img_dir,
                                 augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    return train_loader, val_loader
