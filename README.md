# Fully-Convolutional-Network
Fully Convolutional Neural Network Structure for Image Classification

# Requirements
* Python >= 3.7
* Pytorch >= 1.4.0
* torchvision

## Data
* CIFAR10
* CIFAR100
* Tiny ImageNet
* FaceScrubs
* ImageNet1000
```
python data\dataLoader.py contains pre-processing before training on each data set
```

## Config
```
python conf\config.py contains the configuration of each data set during the training process
```
## Net
* VGG16 and VGG16_FCN
* ResNet50 and ResNet50_FCN
* DenseNet121 and DenseNet121_FCN
* MobileNetV2 and MobileNetV2_FCN
* NASNet and NASNet_FCN
```python network_struture includes the above network```

## Train and Test
```
python train_test\train_test.py
```
