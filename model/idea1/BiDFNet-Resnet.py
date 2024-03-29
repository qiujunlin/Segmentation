
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.Res2Net import  res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1 ,userelu=False):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.userelu =False

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,doubleconv=True):
        super(DecoderBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding)

        self.conv2 = BasicConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding)
        self.doubleconv =doubleconv
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        if self.doubleconv:
           x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AFM(nn.Module):
    def __init__(self, in_channels, all_channels):
        super(AFM, self).__init__()
        self.selayer = SELayer(all_channels)

    def forward(self, higerencoder, encoder, decoder):
        fuse = torch.cat([encoder, decoder, higerencoder], dim=1)
        fuse = self.selayer(fuse)
        return fuse


class RCM(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(RCM, self).__init__()
       # self.conv1 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = BasicConv2d(in_channels, out_channel, 1)
        self.conv3 = BasicConv2d(out_channel, out_channel, 3, padding=1)

    def forward(self, encoder, afm):
      #  encoder = self.conv1(encoder)
        encoder = self.conv2(encoder)
        fuse = encoder + afm
        fuse = self.conv3(fuse)
        return fuse
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class BiDFNet(nn.Module):
    def __init__(self, channel=64):
        super(BiDFNet, self).__init__()

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.Translayer0 = RFB_modified(64, channel)
        self.Translayer1 = RFB_modified(256, channel)
        self.Translayer2 = RFB_modified(512, channel)
        self.Translayer3 = RFB_modified(1024, channel)
        self.Translayer4 = RFB_modified(2048, channel)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=channel, out_channels=channel)
        self.decoder3 = DecoderBlock(in_channels=channel * 3, out_channels=channel)
        self.decoder2 = DecoderBlock(in_channels=channel * 3, out_channels=channel)
        self.decoder1 = DecoderBlock(in_channels=channel * 3, out_channels=channel)
        self.decoder0 = nn.Sequential(BasicConv2d(channel * 3, channel, 1),
                                      BasicConv2d(channel, channel, 1))
        self.decoder5 = nn.Sequential(BasicConv2d(channel * 2, channel, 1),
                                      BasicConv2d(channel, channel, 1))
        self.decoder6 = DecoderBlock(in_channels=channel * 2, out_channels=channel,doubleconv=False)
        self.decoder7 = DecoderBlock(in_channels=channel * 2, out_channels=channel,doubleconv=False)
        self.decoder8 = DecoderBlock(in_channels=channel * 2, out_channels=channel,doubleconv=False)
        self.decoder9 = DecoderBlock(in_channels=channel, out_channels=channel,doubleconv=False)

        # adaptive Flusion module
        self.afm3 = AFM(channel, channel * 3)
        self.afm2 = AFM(channel, channel * 3)
        self.afm1 = AFM(channel, channel * 3)
        self.afm0 = AFM(channel, channel * 3)
        self.afm4 = SELayer(channel)

        self.rcm1 = RCM(channel * 3, channel)
        self.rcm2 = RCM(channel * 3, channel)
        self.rcm3 = RCM(channel * 3, channel)
        self.rcm0 = RCM(channel * 3, channel)
        self.rcm4 = RCM(channel, channel)

        self.unetout1 = nn.Conv2d(channel, 1, 1)
        self.unetout2 = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        basic = x
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x) # 64 172 172

        # ---- low-level features ----
        x = self.resnet.maxpool(x0)  # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88

        # ---- high-level features ----
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        x0 = self.Translayer0(x0)
        x1 = self.Translayer1(x1)
        x2 = self.Translayer2(x2)
        x3 = self.Translayer3(x3)
        x4 = self.Translayer4(x4)

        # decoder1

        afm4 = self.afm4(x4)
        d1_4 = self.decoder4(afm4)  # b 320 22 22
        afm3 = self.afm3(x3, self.upsample(x4), d1_4)  # 512+320+320
        d1_3 = self.decoder3(afm3)  # b 128 44 4
        afm2 = self.afm2(x2, self.upsample(x3), d1_3)
        d1_2 = self.decoder2(afm2)  # b 128 88 88
        afm1 = self.afm1(x1, self.upsample(x2), d1_2)
        d1_1 = self.decoder1(afm1)
        afm0 = self.afm1(x0, self.upsample(x1), d1_1)
        d1_0 = self.decoder0(afm0)

        # rcm
        x0 = self.rcm1(afm0, x0)  # b 64 88 88
        x1 = self.rcm1(afm1, x1)  # b 64 88 88
        x2 = self.rcm2(afm2, x2)  # b 64 44 44
        x3 = self.rcm3(afm3, x3)  # b 64  22 22
        x4 = self.rcm4(afm4, x4)  # b 64 11 11

        # feadback
        guidance = d1_0
        guidance0 = F.interpolate(guidance, scale_factor=1 / 16, mode='bilinear')
        guidance1 = F.interpolate(guidance, scale_factor=1 / 8, mode='bilinear')
        guidance2 = F.interpolate(guidance, scale_factor=1 / 4, mode='bilinear')
        guidance3 = F.interpolate(guidance, scale_factor=1 / 2, mode='bilinear')
        x4 = x4 + guidance0
        x3 = x3 + guidance1
        x2 = x2 + guidance2
        x1 = x1 + guidance3
        x0 = x0 + guidance

        # decoder 2
        x4_1 = x4
        x3_1 = self.upsample(x4) * x3
        x2_1 = self.upsample(x3) * x2
        x1_1 = self.upsample(x2) * x1
        x0_1 = self.upsample(x1) * x0

        x4_1 = self.decoder9(x4_1)
        x3_2 = torch.cat((x3_1, x4_1), 1)
        x3_2 = self.decoder8(x3_2)
        x2_2 = torch.cat((x2_1, x3_2), 1)
        x2_2 = self.decoder7(x2_2)
        x1_2 = torch.cat((x1_1, x2_2), 1)
        x1_2 = self.decoder6(x1_2)

        x0_2 = torch.cat((x0_1, x1_2), 1)
        x0_2 = self.decoder5(x0_2)

        pred1 = self.unetout1(d1_0)
        pred2 = self.unetout2(x0_2)

        pred2 = F.interpolate(pred2, scale_factor=2, mode='bilinear')
        pred1 = F.interpolate(pred1, scale_factor=2, mode='bilinear')

        return pred1, pred2


if __name__ == '__main__':
    model = BiDFNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    pred2, pred1 = model(input_tensor)
    print(pred2.size())
    print(pred1.size())
    # print(prediction1.size())
    # print(prediction2.size())
    # print(prediction3.size())
    # print(prediction4.size())

    # net =BCA(64,64,64)
    # a =torch.rand(1,64,44,44)
    # b =torch.rand(1,64,44,44)
    # print(net(a,b).size())