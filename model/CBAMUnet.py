
import torch
import torch.nn as nn
from model.resnet import resnet34
from torch.nn import functional as F
#import torchsummary
from torch.nn import init
from model.module.attention import  CBAM
class CBAMUnet(nn.Module):
    def __init__(self, out_planes=1, ccm=True, norm_layer=nn.BatchNorm2d, is_training=True, expansion=2,
                 base_channel=32):
        super(CBAMUnet, self).__init__()

        self.backbone = resnet34(pretrained=True)
        self.expansion = expansion
        self.base_channel = base_channel
        if self.expansion == 4 and self.base_channel == 64:
            expan = [512, 1024, 2048]
            spatial_ch = [128, 256]
        elif self.expansion == 4 and self.base_channel == 32:
            expan = [256, 512, 1024]
            spatial_ch = [32, 128]
            conv_channel_up = [256, 384, 512]
        elif self.expansion == 2 and self.base_channel == 32:
            expan = [128, 256, 512]
            spatial_ch = [64, 64]
            conv_channel_up = [128, 256, 512]

        conv_channel = expan[0]

        self.is_training = is_training

        self.decoder5 = DecoderBlock(expan[-1], expan[-2], relu=False, last=True)  # 256
        self.decoder4 = DecoderBlock(expan[-2], expan[-3], relu=False)  # 128
        self.decoder3 = DecoderBlock(expan[-3], spatial_ch[-1], relu=False)  # 64
        self.decoder2 = DecoderBlock(spatial_ch[-1], spatial_ch[-2])  #  32


        self.cbam4 = CBAM(expan[-2])
        self.cbam3=CBAM(expan[-3])
        self.cbam2=CBAM(64)
        self.cbam1=CBAM(64)



        self.main_head = BaseNetHead(spatial_ch[0], out_planes, 2,
                                     is_aux=False, norm_layer=norm_layer)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2  64

        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)  # 1/4   64
        c3 = self.backbone.layer2(c2)  # 1/8   128
        c4 = self.backbone.layer3(c3)  # 1/16   256
        c5 = self.backbone.layer4(c4)  # 1/32   512

        # d_bottom=self.bottom(c5)
        # c5=self.sap(c5)

        # d5=d_bottom+c5           #512

        d4 = self.relu(self.decoder5(c5) + self.cbam4(c4))  # 256
        d3 = self.relu(self.decoder4(d4) + self.cbam3(c3))  # 128
        d2 = self.relu(self.decoder3(d3) + self.cbam2(c2))  # 64
        d1 = self.decoder2(d2) + self.cbam1(c1)  # 32
        main_out = self.main_head(d1)
       # main_out = F.log_softmax(main_out, dim=1)

        # (1,2,448,448)
        return main_out


class DecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(DecoderBlock, self).__init__()

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.scale = scale
        self.last = last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last == False:
            x = self.conv_3x3(x)
            # x=self.sap(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv_1x1(x)
        return x


class BaseNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BaseNetHead, self).__init__()
        if is_aux:
            self.conv_1x1_3x3 = nn.Sequential(
                ConvBnRelu(in_planes, 64, 1, 1, 0,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False),
                ConvBnRelu(64, 64, 3, 1, 1,
                           has_bn=True, norm_layer=norm_layer,
                           has_relu=True, has_bias=False))
        else:
            self.conv_1x1_3x3 = nn.Sequential(
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
if __name__ == '__main__':
    net = CBAMUnet()
    A =  torch.rand((1,3,224,224))
    print(net(A).shape)