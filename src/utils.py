import os
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    transforms.ToTensor()
    ])

def load_data(data_dir, train_size):
    train_data_path = data_dir + "Train"
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform = data_transforms)
    train_len = int(len(train_data) *train_size)
    valid_len = len(train_data) - train_len

    trainset, valset = data.random_split(train_data, [train_len, valid_len])

    return trainset, valset

def data_loader(trainset, validset, batch_size, shuffle):

    trainloader = DataLoader(trainset, shuffle=shuffle, batch_size = batch_size)
    valloader = DataLoader(validset, shuffle=shuffle, batch_size = batch_size)

    return trainloader, valloader

def load_test_data(data_dir):
    test_data_path = data_dir + "tt"
    testset = torchvision.datasets.ImageFolder(root = test_data_path, transform = data_transforms)
    return testset

def test_data_loader(testset):
    testdataloader = DataLoader(testset, shuffle=False, batch_size=1)
    return testdataloader