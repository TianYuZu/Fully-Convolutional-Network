from datetime import datetime
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mpl_toolkits.mplot3d as p3d
import math
import os
import torch.optim as optim
from scipy.special import binom
import scipy.io as io
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import conf.config as conf

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device_ids = [0, 1]


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def get_acc_top5(output, label):
    total = output.shape[0]
    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    correct_k = correct[:5].view(-1).float().sum(0)
    return correct_k / total


def draw(history):
    epochs = range(1, len(history['loss_train']) + 1)
    plt.plot(epochs, history['loss_train'], 'blue', label='Training loss')
    plt.plot(epochs, history['loss1_train'], 'green', label='Training loss1')
    plt.plot(epochs, history['loss2_train'], 'yellow', label='Training loss2')
    plt.plot(epochs, history['loss_val'], 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.imsave('E:/acc_and_loss/Training and Validation loss.jpg')
    plt.figure()
    epochs = range(1, len(history['acc_train']) + 1)
    plt.plot(epochs, history['acc_train'], 'b', label='Training acc')
    plt.plot(epochs, history['acc_val'], 'r', label='validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    # plt.imsave('E:/acc_and_loss/Training and validation acc.jpg')
    plt.show()

    
PEDCC_PATH = 'train_test/100_512.pkl'#load PEDCC

def read_pkl():
    f = open(PEDCC_PATH, 'rb')
    a = pickle.load(f)
    f.close()
    return a    

    
class CosineLinear_PEDCC(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear_PEDCC, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
        #self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        map_dict = read_pkl()
        tensor_empty = torch.Tensor([]).cuda()
        for label_index in range(self.out_features):
            tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
        label_40D_tensor = tensor_empty.view(-1, self.in_features).permute(1, 0)
        label_40D_tensor = label_40D_tensor.cuda()
        self.weight.data = label_40D_tensor

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features
        cos_theta = x.mm(w)  # size=(B,Classnum)  x.dot(ww)

        return cos_theta  # size=(B,Classnum,1)

def train_test_fcn(net, train_data, valid_data, cfg, criterion, save_folder, classes_num):
    LR = cfg['LR']
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()

    prev_time = datetime.now()
    history = dict()
    loss_train = []
    loss1_train = []
    loss2_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    for epoch in range(cfg['max_epoch']):
        if epoch in cfg['lr_steps']:
            if epoch != 0:
                LR *= 0.1
            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_acc = 0
        length, num = 0, 0
        length_test = 0
        net = net.train()
        for im, label in tqdm(train_data):
            if torch.cuda.is_available():
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)

            # forward
            # ims = im.shape
            output = net(im)
            loss = criterion(output, label)
            length += output.pow(2).sum().item()
            num += output.shape[0]
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_acc += get_acc(output, label)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)
                loss = criterion2(output, label)
                valid_loss += loss.data
                valid_acc += get_acc(output, label)
                length_test += output.pow(2).sum().item()/im.shape[0]
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f, length: %f, length_test: %f"
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR, train_loss1 / len(train_data),
                   train_loss2 / len(train_data), length/num, length_test/len(valid_data)))
            loss_train.append(train_loss / len(train_data))
            loss1_train.append(train_loss1 / len(train_data))
            loss2_train.append(train_loss2 / len(train_data))
            loss_val.append(valid_loss / len(valid_data))
            acc_train.append(train_acc / len(train_data))
            acc_val.append(valid_acc / len(valid_data))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open(save_folder+'tiny180_200pedcc.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        if (epoch+1) % 10 == 0:
            torch.save(net.module.state_dict(), save_folder + 'FCN' + str(epoch+1) + '_epoch.pth')

    history['loss_train'] = loss_train
    history['loss1_train'] = loss1_train
    history['loss2_train'] = loss2_train
    history['loss_val'] = loss_val
    history['acc_train'] = acc_train
    history['acc_val'] = acc_val
    draw(history)
