# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# different DATA set configs
CIFAR10 = {
    'num_classes': 10,
    'max_epoch': 100,
    'LR': 1e-1,
    'lr_steps': (0, 30, 60, 90),
    'batch_size': 200,
    'name': 'CIFAR10'
}
CIFAR100 = {
    'num_classes': 100,
    'max_epoch': 200,
    'LR': 1e-1,
    'lr_steps': (0, 50, 100, 150),
    'batch_size': 128,
    'name': 'CIFAR100'
}
TinyImageNet = {
    'num_classes': 200,
    'max_epoch': 100,
    'LR': 1e-1,
    'lr_steps': (0, 30, 60, 90),
    'batch_size': 256,
    'name': 'TinyImageNet'
}
FaceScrubs = {
    'num_classes': 100,
    'max_epoch': 100,
    'LR': 1e-1,
    'lr_steps': (0, 30, 60, 90),
    'batch_size': 128,
    'name': 'FaceScrubs'
}
ImageNet1000 = {
    'num_classes': 1000,
    'max_epoch': 20,
    'LR': xxx,   # Set according to the specific situation
    'lr_steps': (xxx, xxx, xxx, xxx),   #Set according to the specific situation
    'batch_size': 192,
    'name': 'ImageNet1000'
}
