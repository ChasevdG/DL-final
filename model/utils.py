import numpy as np
import pystk

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

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
        print('getting item')
        data = self.data[idx]
        data = self.transform(*data)
        return data
    
class DistDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        from PIL import Image
        from glob import glob
        from os import path
        self.data = []
        for f in glob(path.join(dataset_path, '*.csv')):
            i = Image.open(f.replace('.csv', ''))
            i.load()
            labels = np.loadtxt(f, dtype=np.float32, delimiter=',')
            self.data.append((i, labels[2]))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print('getting item')
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_dist_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = DistDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
