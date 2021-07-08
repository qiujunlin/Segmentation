import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class Residual_Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y


class Squeeze_Excite(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(channel // reduction, channel, bias=False),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_enc=True, residual=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.SE = Squeeze_Excite(out_channels, 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = Residual_Shortcut(in_channels, out_channels)

        self.is_enc = is_enc
        self.residual = residual

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.SE(y)

        if self.residual:
            y = y + self.shortcut(x)

        if self.is_enc:
            y = self.pool(y)

        return y


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)

        self.conv5 = nn.Conv2d(self.out_channels * 5, self.out_channels, kernel_size=1)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        y_pool0 = nn.AdaptiveAvgPool2d(output_size=1)(x)
        y_conv0 = self.conv0(y_pool0)
        y_conv0 = self.bn(y_conv0)
        y_conv0 = self.relu(y_conv0)
        y_conv0 = nn.Upsample(size=(h, w), mode='bilinear')(y_conv0)

        y_conv1 = self.conv1(x)
        y_conv1 = self.bn(y_conv1)
        y_conv1 = self.relu(y_conv1)

        y_conv2 = self.conv2(x)
        y_conv2 = self.bn(y_conv2)
        y_conv2 = self.relu(y_conv2)

        y_conv3 = self.conv3(x)
        y_conv3 = self.bn(y_conv3)
        y_conv3 = self.relu(y_conv3)

        y_conv4 = self.conv4(x)
        y_conv4 = self.bn(y_conv4)
        y_conv4 = self.relu(y_conv4)

        y = torch.cat([y_conv0, y_conv1, y_conv2, y_conv3, y_conv4], 1)
        y = self.conv5(y)
        y = self.bn(y)
        y = self.relu(y)

        return y


def output_block():
    Layer = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1)),
                          nn.Sigmoid())
    return Layer


class DoubleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1_1 = VGGBlock(3, 64, 64, True)
        self.enc1_2 = VGGBlock(64, 128, 128, True)
        self.enc1_3 = VGGBlock(128, 256, 256, True)
        self.enc1_4 = VGGBlock(256, 512, 512, True)
        self.enc1_5 = VGGBlock(512, 512, 512, True)

        # apply pretrained vgg19 weights on 1st unet
        vgg19 = models.vgg19_bn()
        vgg19.load_state_dict(torch.load("E:\dataset\pretrain/vgg19_bn-c79401a0.pth"))

        self.enc1_1.conv1.weights = vgg19.features[0].weight
        self.enc1_1.bn1.weights = vgg19.features[1].weight
        self.enc1_1.conv2.weights = vgg19.features[3].weight
        self.enc1_1.bn2.weights = vgg19.features[4].weight
        self.enc1_2.conv1.weights = vgg19.features[7].weight
        self.enc1_2.bn1.weights = vgg19.features[8].weight
        self.enc1_2.conv2.weights = vgg19.features[10].weight
        self.enc1_2.bn2.weights = vgg19.features[11].weight
        self.enc1_3.conv1.weights = vgg19.features[14].weight
        self.enc1_3.bn1.weights = vgg19.features[15].weight
        self.enc1_3.conv2.weights = vgg19.features[17].weight
        self.enc1_3.bn2.weights = vgg19.features[18].weight
        self.enc1_4.conv1.weights = vgg19.features[27].weight
        self.enc1_4.bn1.weights = vgg19.features[28].weight
        self.enc1_4.conv2.weights = vgg19.features[30].weight
        self.enc1_4.bn2.weights = vgg19.features[31].weight
        self.enc1_5.conv1.weights = vgg19.features[33].weight
        self.enc1_5.bn1.weights = vgg19.features[34].weight
        self.enc1_5.conv2.weights = vgg19.features[36].weight
        self.enc1_5.bn2.weights = vgg19.features[37].weight
        del vgg19

        self.aspp1 = ASPP(512, 512)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec1_4 = VGGBlock(1024, 256, 256, False)
        self.dec1_3 = VGGBlock(512, 128, 128, False)
        self.dec1_2 = VGGBlock(256, 64, 64, False)
        self.dec1_1 = VGGBlock(128, 32, 32, False)

        self.output1 = output_block()

        self.enc2_1 = VGGBlock(3, 64, 64, True, True)
        self.enc2_2 = VGGBlock(64, 128, 128, True, True)
        self.enc2_3 = VGGBlock(128, 256, 256, True, True)
        self.enc2_4 = VGGBlock(256, 512, 512, True, True)
        self.enc2_5 = VGGBlock(512, 512, 512, True, True)

        self.aspp2 = ASPP(512, 512)

        self.dec2_4 = VGGBlock(1536, 256, 256, False, True)
        self.dec2_3 = VGGBlock(768, 128, 128, False, True)
        self.dec2_2 = VGGBlock(384, 64, 64, False, True)
        self.dec2_1 = VGGBlock(192, 32, 32, False, True)

        self.output2 = output_block()

    def forward(self, _input):
        # encoder of 1st unet
        y_enc1_1 = self.enc1_1(_input)
        y_enc1_2 = self.enc1_2(y_enc1_1)
        y_enc1_3 = self.enc1_3(y_enc1_2)
        y_enc1_4 = self.enc1_4(y_enc1_3)
        y_enc1_5 = self.enc1_5(y_enc1_4)

        # aspp bridge1
        y_aspp1 = self.aspp1(y_enc1_5)

        # decoder of 1st unet
        y_dec1_4 = self.up(y_aspp1)
        y_dec1_4 = self.dec1_4(torch.cat([y_enc1_4, y_dec1_4], 1))
        y_dec1_3 = self.up(y_dec1_4)
        y_dec1_3 = self.dec1_3(torch.cat([y_enc1_3, y_dec1_3], 1))
        y_dec1_2 = self.up(y_dec1_3)
        y_dec1_2 = self.dec1_2(torch.cat([y_enc1_2, y_dec1_2], 1))
        y_dec1_1 = self.up(y_dec1_2)
        y_dec1_1 = self.dec1_1(torch.cat([y_enc1_1, y_dec1_1], 1))
        y_dec1_0 = self.up(y_dec1_1)

        # output of 1st unet
        output1 = self.output1(y_dec1_0)

        # multiply input and output of 1st unet
        mul_output1 = _input * output1

        # encoder of 2nd unet
        y_enc2_1 = self.enc2_1(mul_output1)
        y_enc2_2 = self.enc2_2(y_enc2_1)
        y_enc2_3 = self.enc2_3(y_enc2_2)
        y_enc2_4 = self.enc2_4(y_enc2_3)
        y_enc2_5 = self.enc2_5(y_enc2_4)

        # aspp bridge 2
        y_aspp2 = self.aspp2(y_enc2_5)

        # decoder of 2nd unet
        y_dec2_4 = self.up(y_aspp2)
        y_dec2_4 = self.dec2_4(torch.cat([y_enc1_4, y_enc2_4, y_dec2_4], 1))
        y_dec2_3 = self.up(y_dec2_4)
        y_dec2_3 = self.dec2_3(torch.cat([y_enc1_3, y_enc2_3, y_dec2_3], 1))
        y_dec2_2 = self.up(y_dec2_3)
        y_dec2_2 = self.dec2_2(torch.cat([y_enc1_2, y_enc2_2, y_dec2_2], 1))
        y_dec2_1 = self.up(y_dec2_2)
        y_dec2_1 = self.dec2_1(torch.cat([y_enc1_1, y_enc2_1, y_dec2_1], 1))
        y_dec2_0 = self.up(y_dec2_1)

        # output of 2nd unet
        output2 = self.output2(y_dec2_0)

        return output1, output2

if __name__ == '__main__':
    net = DoubleUNet()
    a= torch.rand((1,3,224,224))
    print(net(a))