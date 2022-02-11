import torch

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler

dat = {"MNIST" : MNIST}
def get_dataloader(dataset_name, batch_size,num_workers):
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,),(0.5,))
                    ])
    train = dat[dataset_name](
        root = './data/',
        train = True, 
        download = True, 
        transform = transform
        )
    dataloader = DataLoader(train,sampler=RandomSampler(train),batch_size=batch_size,num_workers=num_workers,drop_last=True)
    return dataloader