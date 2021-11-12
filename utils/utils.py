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

def val_softmax(args, model, dataloader):
    print('\n')
    print('Start Validation!')
    with torch.no_grad():  # 在评价过程中停止求梯度  加快速度的作用
        model.eval()  # !!!评价函数必须使用
        tbar = tqdm.tqdm(dataloader, desc='\r')
        total_Dice = []
        dice1 = []
        total_Dice.append(dice1)
        Acc = []
        cur_cube = []
        cur_label_cube = []
        for i, (data, labels) in enumerate(tbar):
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = labels[0].cuda()
            slice_num = labels[1].long().item()  # 获取总共的label数量  86
            # get RGB predict image
            predicts = model(data)  # 预测结果  经过softmax后的 float32
            predict = torch.argmax(torch.exp(predicts), dim=1)  # int64 # n h w 获取的是结果 预测的结果是属于哪一类的
            batch_size = predict.size()[0]  # 当前的批量大小   1
            cur_cube.append(predict)  # (1,h,w)
            cur_label_cube.append(label)  #
        predict_cube = torch.stack(cur_cube, dim=0).squeeze()  # (n,h,w) int 64 tensor
        label_cube = torch.stack(cur_label_cube, dim=0).squeeze()  # n hw float32 tensor
        assert predict_cube.size()[0] == slice_num
        # 计算
        Dice, acc = eval_multi_seg(predict_cube, label_cube, args.num_classes)
        for class_id in range(args.num_classes - 1):
                total_Dice[class_id].append(Dice[class_id])
        Acc.append(acc)
        dice1 = sum(total_Dice[0])
        ACC = sum(Acc) / len(Acc)
        tbar.set_description('Dice1: %.3f,ACC: %.3f' % (dice1, ACC))
        print('Dice1:', dice1)
        print('Acc:', ACC)
        return dice1, ACC
"""
评价sigmod的函数
"""
def val_sigmod(args, model, dataloader):
    print('\n')
    print('Start Validation!')
    with torch.no_grad():  # 在评价过程中停止求梯度  加快速度的作用
        model.eval()  # !!!评价函数必须使用
        tbar = tqdm.tqdm(dataloader, desc='\r')
        cur_cube=[]
        cur_label_cube=[]
        counter=0
        for i, (data, labels) in enumerate(tbar):
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = labels[0].cuda()
            slice_num = labels[1].long().item()  # 获取总共的label数量  86
            # get RGB predict image
            predicts= model(data)  # 预测结果  经过sigmod后的 float32
            predict = (predicts[-1]>0.5).float()  # int64 # n h w 获取的是结果 预测的结果是属于哪一类的
            batch_size = predict.size()[0]  # 当前的批量大小   1
            counter += batch_size  # 每次加一
            cur_cube.append(predict)  # (1,h,w)
            cur_label_cube.append(label)  #
        predict_cube = torch.stack(cur_cube, dim=0).squeeze()  # (n,h,w) int 64 tensor
        label_cube = torch.stack(cur_label_cube, dim=0).squeeze()  # n hw float32 tensor
        # 计算
        Dice,acc = eval_sseg(predict_cube, label_cube)
        tbar.set_description('Dice1: %.3f,ACC: %.3f' % (Dice, acc))
        print('Dice1:', Dice)
        print('Acc:', acc)
        return Dice, acc

def eval_sseg(predict, target):
    """
       返回多分类的损失函数
       """
    # pred_seg=torch.argmax(torch.exp(predict),dim=1).int()
    pred_seg = predict.data.cpu().numpy()  # n h w int64 ndarray
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)  # n h w float32  -> int32 ndarray
    assert (pred_seg.shape == label_seg.shape)
    acc = (pred_seg == label_seg).sum() / (
                pred_seg.shape[0] * pred_seg.shape[1] * pred_seg.shape[2])  # acc 就是所有相同的像素值占总像素的大小
    overlap = ((pred_seg == 1) * (label_seg == 1)).sum()
    union = (pred_seg == 1).sum() + (label_seg == 1).sum()
    Dice=((2 * overlap + 0.1) / (union + 0.1))
    return Dice, acc


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



