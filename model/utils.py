import numpy as np
import pystk

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms
from PIL import Image

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = 'drive_data'

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', ''))
            i.load()
            labels = np.loadtxt(f, dtype=np.float32, delimiter=',')
            self.data.append((i, labels[0:2]))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data
    
class DistDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        print('Loading Data')
        for f in glob(path.join(dataset_path, '*.csv')):
            img = f.replace('.csv', '')
            labels = np.loadtxt(f, dtype=np.float32, delimiter=',')
            self.data.append((img, labels[2]))
        print('Finished Loading Data')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        f = data[0]
        i = Image.open(f)
        i.load()
        data = (self.transform(i),data[1])
        return data

class On_Screen_Dataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        print('Loading Data') 
        w_path = dataset_path + '/with_Ball'
        for f in glob(path.join(w_path, '*.csv')):
            self.data.append((f, 1))
        print('Finished Loading with Ball Data') 
        wo_path = dataset_path + '/without_Ball'
        for f in glob(path.join(wo_path, '*.png')):
            self.data.append((f, 0))
        print('Finished Loading Data') 
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        f = data[0]
        if(data[1]):
            i = Image.open(f.replace('.csv', ''))
        else:
            i = Image.open(f)
        i.load()
        data = (self.transform(i),data[1])
        return data

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_dist_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = DistDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_on_screen_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = On_Screen_Dataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_on_screen_data_split(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = On_Screen_Dataset(dataset_path, transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    return DataLoader(train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True), DataLoader(val_set, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
