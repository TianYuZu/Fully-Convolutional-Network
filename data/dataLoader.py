import torchvision
import torchvision.transforms as transforms
import os
import torch
import cv2
import torch.utils.data as data
from PIL import Image

# You can find these data sets online
CIFAR10_Train_ROOT = xxx
CIFAR10_Test_ROOT = xxx
CIFAR100_Train_ROOT = xxx
CIFAR100_Test_ROOT = xxx
TinyImageNet_Train_ROOT = xxx
TinyImageNet_Test_ROOT = xxx
FaceScrubs_Train_ROOT = xxx
FaceScrubs_Test_ROOT = xxx
ImageNet1000_Train_ROOT = xxx
ImageNet1000_Test_ROOT = xxx


#CIFAR10 data set
CIFAR10_train_data = torchvision.datasets.ImageFolder(CIFAR10_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)
CIFAR10_test_data = torchvision.datasets.ImageFolder(CIFAR10_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)

# CIFAR100 data set
CIFAR100_train_data = myImageFloder(CIFAR100_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
CIFAR100_test_data = torchvision.datasets.ImageFolder(CIFAR100_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
)

# TinyImageNet data set
TinyImageNet_train_data = torchvision.datasets.ImageFolder(TinyImageNet_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
TinyImageNet_test_data = torchvision.datasets.ImageFolder(TinyImageNet_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)

# FaceScrubs data set
FaceScrubs_train_data = torchvision.datasets.ImageFolder(FaceScrubs_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)
Facescrubs_test_data = torchvision.datasets.ImageFolder(Facescrubs_Test_ROOT,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])
)

# ImageNet1000 data set
ImageNet1000_train_data = torchvision.datasets.ImageFolder(ImageNet1000_Train_ROOT,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)
ImageNet1000_test_data = torchvision.datasets.ImageFolder(ImageNet1000_Test_ROOT,
    transform=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)