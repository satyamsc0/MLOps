import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import warnings

class DeviceDataLoader():
    """Wrapper around dataloaders to transfer batches to specified devices"""
    def __init__(self, dataloader, device):
        self.dl = dataloader
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

def denormalize(images, means, std_devs):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    std_devs = torch.tensor(std_devs).reshape(1, 3, 1, 1)
    return images * std_devs + means

def show_batch(dl, data_statistics):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(10, 10))
        images = denormalize(images, *data_statistics)
        ax.imshow(make_grid(images, 10).permute(1,2,0)) # HxWxC
        break

def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_device(entity, device):
    if isinstance(entity, (list, tuple)):
        return [to_device(ele, device) for ele in entity]
    return entity.to(device, non_blocking=True)

def preprocess_data():
    warnings.filterwarnings("ignore")
    
    data_statistics = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # can be calculated from our dataset as well

    train_transforms_cifar = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'), # augmentation
        transforms.RandomHorizontalFlip(), # augmentation
        transforms.ToTensor(), # CxHxW
        transforms.Normalize(*data_statistics, inplace=True) # [-1,1], data = (data - mean)/std_dev
    ])

    test_transforms_cifar = transforms.Compose([
        transforms.ToTensor(), # CxHxW
        transforms.Normalize(*data_statistics, inplace=True) # [-1,1], data = (data - mean)/std_dev
    ])

    dataset = torchvision.datasets.CIFAR10(root='data/', download=True, transform=train_transforms_cifar)
    dataset = torchvision.datasets.CIFAR10(root='data/', download=True, train=False, transform=test_transforms_cifar)

    val_ratio = 0.2 # validation data ratio
    train_dataset, val_dataset = random_split(dataset, [int((1-val_ratio)*len(dataset)), int(val_ratio*len(dataset))])

    batch_size = 64
    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=True, pin_memory=True)
    test_dl = DataLoader(val_dataset, batch_size, pin_memory=True)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    return train_dl, val_dl, test_dl, device

if __name__ == "__main__":
    train_dl, val_dl, test_dl, device = preprocess_data()
    show_batch(train_dl, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
