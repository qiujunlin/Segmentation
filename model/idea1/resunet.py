# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""

import torch
import torch.nn as nn
from model import resnet50
from torch.nn import functional as F
from torch.nn import init
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class Resunet(nn.Module):
    def __init__(self,out_planes=1,ccm=True,norm_layer=nn.BatchNorm2d,is_training=True,expansion=2,base_channel=32):
        super(Resunet,self).__init__()
        
        self.backbone =resnet50(pretrained=True)
        self.expansion=expansion
        self.base_channel=base_channel
        if self.expansion==2 and self.base_channel==32:
            expan=[64,256,512,1024,2048]

        self.is_training = is_training
        self.decoder5=DecoderBlock(expan[-1],expan[-2],relu=False,last=True) #256
        self.decoder4=DecoderBlock(expan[-2],expan[-3],relu=False) #128
        self.decoder3=DecoderBlock(expan[-3],expan[-4],relu=False) #64
        self.decoder2=DecoderBlock(expan[-4],expan[-5]) #32

        self.main_head= BaseNetHead(expan[0], out_planes, 2,
                             is_aux=False, norm_layer=norm_layer)
       
        self.relu = nn.ReLU()

    def forward(self, x):
        
        
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)#1/2  64
        
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)#1/4   64
        c3 = self.backbone.layer2(c2)#1/8   128
        c4 = self.backbone.layer3(c3)#1/16   256
        c5 = self.backbone.layer4(c4)#1/32   512

        d4=self.relu(self.decoder5(c5,c4))  #256
        d3=self.relu(self.decoder4(d4,c3) ) #128
        d2=self.relu(self.decoder3(d3,c2)) #64
        d1=self.decoder2(d2,c1) #32
        main_out=self.main_head(d1)

        return main_out
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
#        return F.logsigmoid(main_out,dim=1)




class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3=nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3=nn.Sequential(
                ConvBnRelu(in_planes, 32, 1, 1, 0,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False),
                ConvBnRelu(32, 32, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False))
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1_2 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1_2 = nn.Conv2d(32, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)
        fm = self.conv_1x1_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1_2(fm)
        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d,scale=2,relu=True,last=False):
        super(DecoderBlock, self).__init__()


        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes+out_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

        self.scale=scale
        self.last=last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x,y):

        if self.last==False:
            x = self.conv_3x3(x)
            # x=self.sap(x)
        if self.scale>1:
            x=F.interpolate(x,scale_factor=self.scale,mode='bilinear',align_corners=True)
        x = torch.cat([x,y],dim=1)
        x=self.conv_1x1(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs




if __name__ == '__main__':
    from torchsummary import summary
    model = Resunet(out_planes=2)
    model = model.cuda()
    a= torch.rand((1,3,448,448)).cuda()
    print(model(a))
    from torchsummary import summary

    summary(model, input_size=(3, 352, 352))
