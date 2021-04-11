import  os
import  torch
import numpy as np
import  torch.nn as nn
import  torchvision.models as models
from resnet import  resnet50
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
def a():
    num = torch.rand((1,3,224,224))
    num =  torch.max(num,dim=1,keepdim=True)
    print(num[0].shape)
def test():
    net   =  SpatialAttention(kernel_size=3)
    ar   = torch.rand((1,3,224,224))
    print(net(ar))
    print(net(ar))
def avg():
    net = nn.AdaptiveAvgPool2d(1)
    num = torch.rand((1, 3, 224, 224))
    print(net(num).shape)
def model():
    resnet = models.resnet50(pretrained=True)
    resnetdict  = resnet.state_dict()
    net =  resnet50(pretrained=True)
    netdit  = net.state_dict()
    # for k in netdit.keys():
    #     print(k)
    for k in resnetdict.keys():
        print(k)

model()