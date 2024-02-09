from torchvision import datasets
from torch.utils.data import random_split
import torch

data_dir = 'ferplus_u'
train_split = 0.0
dataset = datasets.ImageFolder(data_dir)
train_size = int(train_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
torch.save(train_dataset, f'{data_dir}_train_dataset.pt')
torch.save(test_dataset, f'{data_dir}_test_dataset.pt')