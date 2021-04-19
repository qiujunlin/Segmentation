import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DiceLoss(nn.Module):
    def __init__(self,smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self,input, target):
        input = torch.sigmoid(input)
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
