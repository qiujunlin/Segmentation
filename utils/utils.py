# coding=gbk
#import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch
from torch.nn import functional as F
#from PIL import Image
import numpy as np
import pandas as pd
#import os
import os.path as osp
import shutil
#import math
import tqdm
from math import exp
class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))



def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating mis-alignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def one_hot_it(label, label_info):
	# return semantic_map -> [H, W, num_classes]
	semantic_map = []
	for info in label_info:
		color = label_info[info]
		# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
		equality = np.equal(label, color)
		class_map = np.all(equality, axis=-1)
		semantic_map.append(class_map)
	semantic_map = np.stack(semantic_map, axis=-1)
	return semantic_map


def compute_score(predict, target, forground = 1,smooth=1):
    score = 0
    count = 0
    target[target!=forground]=0
    predict[predict!=forground]=0
    assert(predict.shape == target.shape)
    overlap = ((predict == forground)*(target == forground)).sum() #TP
    union=(predict == forground).sum() + (target == forground).sum()-overlap #FP+FN+TP
    FP=(predict == forground).sum()-overlap #FP
    FN=(target == forground).sum()-overlap #FN
    TN= target.shape[0]*target.shape[1]-union #TN

    #print('overlap:',overlap)
    dice=(2*overlap +smooth)/ (union+overlap+smooth)

    precsion=((predict == target).sum()+smooth) / (target.shape[0]*target.shape[1]+smooth)

    jaccard=(overlap+smooth) / (union+smooth)

    Sensitivity=(overlap+smooth) / ((target == forground).sum()+smooth)

    Specificity=(TN+smooth) / (FP+TN+smooth)


    return dice,precsion,jaccard,Sensitivity,Specificity






def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


class DiceLoss(nn.Module):
    def __init__(self,smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self,input, target):
        # input = torch.sigmoid(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        intersect=(input*target).sum()
        union = torch.sum(input) + torch.sum(target)
        Dice=(2*intersect+self.smooth)/(union+self.smooth)
        dice_loss=1-Dice
        return dice_loss

class Multi_DiceLoss(nn.Module):

    def __init__(self, class_num=5,smooth=0.001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
    def forward(self,input, target):
        input = torch.exp(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(0,self.class_num):
            input_i = input[:,i,:,:]
            target_i = (target == i).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice/(self.class_num)
        return dice_loss

class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=2,smooth=1,gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self,input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1,self.class_num):
            input_i = input[:,i,:,:]
            target_i = (target == i).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice))**self.gamma
        dice_loss = Dice/(self.class_num - 1)
        return dice_loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.nll_loss(inputs, targets, reduce=False)
        else:
            BCE_loss = F.nll_loss(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



# ------- 1. define loss function --------

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b


def gaussian(window_size, sigma):
 gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
 return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = SSIM(window_size=11,size_average=True)
iou_loss = IOU(size_average=True)


def bce_ssim_loss(pred,target):
    pred=F.sigmoid(pred)
   # target =F.sigmoid(target)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)
    loss =  ssim_out + iou_out
    return loss

if __name__ == '__main__':
    a =torch.randn(1,3,256,256)
    b =torch.randn(1,3,256,256)
    print(bce_ssim_loss(a,b))