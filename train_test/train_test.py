import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import conf.config as conf
import utils
import argparse
from data.dataLoader import *
from network_struture import *

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='FCN Training With Pytorch')
parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'FaceScrubs', 'ImageNet1000', 'miniImageNet'],
                    type=str, help='CIFAR10, CIFAR100, TinyImageNet, FaceScrubs or ImageNet1000')
parser.add_argument('--dataset_root', default='',
                    help='Dataset root directory path')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--save_folder', default='./models/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'CIFAR10':
        train_data = CIFAR10_train_data
        test_data = CIFAR10_test_data
        cfg = conf.CIFAR10

    elif args.dataset == 'CIFAR100':
        train_data = CIFAR100_train_data
        test_data = CIFAR100_test_data
        cfg = conf.CIFAR100

    elif args.dataset == 'TinyImageNet':
        train_data = TinyImageNet_train_data
        test_data = TinyImageNet_test_data
        cfg = conf.TinyImageNet

    elif args.dataset == 'FaceScrubs':
        train_data = FaceScrubs_train_data
        test_data = FaceScrubs_test_data
        cfg = conf.FaceScrubs

    elif args.dataset == 'ImageNet1000':
        train_data = ImageNet1000_train_data
        test_data = ImageNet1000_test_data
        cfg = conf.ImageNet1000
        
    elif args.dataset == 'miniImageNet':
        train_data = miniImageNet_train_data
        test_data = miniImageNet_test_data
        cfg = conf.miniImageNet

    else:
        print("dataset doesn't exist!")
        exit(0)

    # VGG16 and VGG16_FCN
    cnn = VGG16(features=VGG16.make_layers(VGG16.cfgs['D'], batch_norm=True), num_classes=cfg['num_classes'])
    # cnn = VGG16_FCN(features=VGG16_FCN.make_layers(VGG16_FCN.cfgs['D'], batch_norm=True), num_classes=cfg['num_classes'])
    # ResNet50 and ResNet50_FCN
    # cnn = ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=cfg['num_classes'])
    # cnn = ResNet50_FCN(Bottleneck, [3, 4, 6, 3], num_classes=cfg['num_classes'])
    # cnn = ResNet50_FCN_Plus_POD_Loss(Bottleneck, [3, 4, 6, 3], num_classes=cfg['feature_sizes'])  #modify the output of the last convolutional layer to the feature dimension of PEDCC.
    # DenseNet121 and DenseNet121_FCN
    # cnn = DenseNet121(num_classes=cfg['num_classes'])
    # cnn = DenseNet121_FCN(num_classes=cfg['num_classes'])
    # MobileNetV2 and MobileV2_FCN
    # cnn = MobileNetV2(n_class=cfg['num_classes'], input_size=32, width_mult=1.)
    # cnn = MobileNetV2(n_class=cfg['num_classes'], input_size=32, width_mult=1.)
    # NASNet and NASNet_FCN
    # cnn = NASNet(4, 2, 44, 44)  # For CIFAR100, the number of classes can only be 100
    # cnn = NASNet_FCN(4, 2, 44, 44)  # For CIFAR100, the number of classes can only be 100

    criterion = nn.CrossEntropyLoss()
    

    # train_loader = data.DataLoader(dataset, args.batch_size,num_workers=args.num_workers,shuffle=True, collate_fn=detection_collate, pin_memory=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=cfg['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=cfg['batch_size'], shuffle=False, num_workers=6, pin_memory=True)

    # start training
    utils.train_test_fcn(cnn, train_loader, test_loader, cfg, criterion, args.save_folder, cfg['num_classes'])


if __name__ == '__main__':
    train()
