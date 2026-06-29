import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from utils import extract_features
import os

class LensDataset(Dataset):
    def __init__(self, csv_file, base_img_dir, transform=None,
                 filter_clean_data=False, augment=False):
        self.data_frame   = pd.read_csv(csv_file)
        self.base_img_dir = base_img_dir
        self.augment      = augment

        if filter_clean_data:
            initial = len(self.data_frame)
            self.data_frame = self.data_frame[
                self.data_frame['true_class_index'] == self.data_frame['resnet_predicted_index']
            ].reset_index(drop=True)
            print(f"Filtered {csv_file}: kept {len(self.data_frame)}, "
                  f"dropped {initial - len(self.data_frame)}")

        # ColorJitter simulates lighting variation — directly relevant
        # to what the model is learning to compensate for
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
        target = int(row['best_param_index'])

        img_path = os.path.join(self.base_img_dir, row['ae_image_path'])
        image    = Image.open(img_path).convert('RGB')

        if self.transform:
            img_tensor = self.transform(image)
        elif self.augment:
            img_tensor = self.train_transform(image)
        else:
            img_tensor = self.val_transform(image)

        # 74-dim feature vector: RGB histograms + clipping + spatial contrast
        # All derived purely from the image — no metadata used
        feature_vector = extract_features(img_tensor)

        return feature_vector, target


def get_dataloader(train_csv, val_csv, base_img_dir, batch_size=64):
    train_dataset = LensDataset(train_csv, base_img_dir,
                                filter_clean_data=False, augment=True)
    val_dataset   = LensDataset(val_csv,   base_img_dir,
                                filter_clean_data=False, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    return train_loader, val_loader
