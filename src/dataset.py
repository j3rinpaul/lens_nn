import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from utils import extract_spatial_histogram
import os

class LensDataset(Dataset):
    def __init__(self,csv_file,base_img_dir,transform=None,filter_clean_data = True):
        self.data_frame = pd.read_csv(csv_file)
        self.base_img_dir = base_img_dir

        if filter_clean_data:
            initial_length = len(self.data_frame)
            self.data_frame = self.data_frame[self.data_frame['true_class_index'] == self.data_frame["resnet_predicition"]].reset_index(drop=True)
            print(f"filtered out {csv_file}: kept {len(self.data_frame)} --> dropped {initial_length - len(self.data_frame)}")

        self.transform = transform if transform else transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()

        csv_path = self.data_frame.iloc[index]['ae_image_path']
        img_path = os.path.join(self.base_img_dir, csv_path)

        target_parameter = int(self.data_frame.iloc[index]['best_param_index'])

        
        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image)

        light_vector = extract_spatial_histogram(img_tensor,grid_size = 2,bins = 10)

        return light_vector, target_parameter
    

def get_dataloader(train_csv,val_csv,base_img_dir,batch_size = 64):
    train_dataset = LensDataset(train_csv,base_img_dir,filter_clean_data=False)
    val_dataset = LensDataset(val_csv,base_img_dir,filter_clean_data=False)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers = 0)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers = 0)

    return train_loader,val_loader

