
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.pvtv2 import pvt_v2_b2


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
                 kernel_size=3, stride=1, padding=1,doubleconv=True,up =True):
        super(DecoderBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding)

        self.conv2 = BasicConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding)
        self.doubleconv =doubleconv
        self.up = up
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        if self.doubleconv:
           x = self.conv1(x)
        x = self.conv2(x)
        if self.up:
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

class FSM(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(FSM, self).__init__()
        self.selayer = SELayer(inchannel)
        self.conv = BasicConv2d(inchannel,outchannel,1)

    def forward(self, higerencoder, encoder):
        fuse = torch.cat([encoder,higerencoder], dim=1)
        fuse = self.selayer(fuse)
        fuse =self.conv(fuse)
        return fuse

class RCM(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(RCM, self).__init__()
        self.conv2 = BasicConv2d(in_channels, out_channel, 1)
        self.conv3 = BasicConv2d(out_channel, out_channel, 3, padding=1)

    def forward(self, afm, encoder):
        afm = self.conv2(afm)
        #fuse = encoder + afm
      #  fuse = self.conv3(in)
        return afm


class BiDFNet(nn.Module):
    def __init__(self, channel=32):
        super(BiDFNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'F:\pretrain\pvt_v2_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer1 = BasicConv2d(64, channel, 1)
        self.Translayer2 = BasicConv2d(128, channel, 1)
        self.Translayer3 = BasicConv2d(320, channel, 1)
        self.Translayer4 = BasicConv2d(512, channel, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=channel, out_channels=channel)
        self.decoder3 = DecoderBlock(in_channels=channel * 2, out_channels=channel)
        self.decoder2 = DecoderBlock(in_channels=channel * 2, out_channels=channel)
        self.decoder1 = DecoderBlock(in_channels=channel*2, out_channels=channel,up=False)
        self.decoder5 = DecoderBlock(in_channels=channel*2,out_channels=channel,doubleconv=True,up = False)
        self.decoder6 = DecoderBlock(in_channels=channel * 2, out_channels=channel,doubleconv=True)
        self.decoder7 = DecoderBlock(in_channels=channel * 2, out_channels=channel,doubleconv=True)
        self.decoder8 = DecoderBlock(in_channels=channel, out_channels=channel,doubleconv=True)

        # adaptive Flusion module
        self.afm3 = FSM (channel * 2,channel)
        self.afm2 = FSM(channel * 2,channel)
        self.afm1 = FSM(channel * 2,channel)
        #self.afm4 = SELayer(channel)
        self.afm5 = FSM(channel * 2,channel)
        self.afm6 = FSM(channel * 2,channel)
        self.afm7 = FSM(channel * 2,channel)


        self.rcm1 = RCM(channel * 2, channel)
        self.rcm2 = RCM(channel * 2, channel)
        self.rcm3 = RCM(channel * 2, channel)
        self.rcm4 = RCM(channel, channel)

        self.unetout1 = nn.Conv2d(channel, 1, 1)
        self.unetout2 = nn.Conv2d(channel, 1, 1)


    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]  # 1 64 88 88
        x2 = pvt[1]  # 1 128 44 44
        x3 = pvt[2]  # 1 320 22 22
        x4 = pvt[3]  # 1 512 11 11
        x1 = self.Translayer1(x1)
        x2 = self.Translayer2(x2)
        x3 = self.Translayer3(x3)
        x4 = self.Translayer4(x4)

        # decoder1

       # afm4 = self.afm4(x4)
        d1_4 = self.decoder4(x4)  # b  22 22
        afm3 = self.afm3(x3, self.upsample(x4))
        fu3 = torch.cat((afm3,d1_4),dim=1)
        d1_3 = self.decoder3(fu3)  # b  44 4
        afm2 = self.afm2(x2, self.upsample(x3))
        fu2 =  torch.cat((afm2,d1_3),dim=1)
        d1_2 = self.decoder2(fu2)  # b  88 88
        afm1 = self.afm1(x1, self.upsample(x2))
        fu1 = torch.cat((afm1, d1_2), dim=1)
        d1_1 = self.decoder1(fu1)

        # rcm
        x1 = self.rcm1(fu1, x1)  # b 64 88 88
        x2 = self.rcm2(fu2, x2)  # b 64 44 44
        x3 = self.rcm3(fu3, x3)  # b 64  22 22
      #  x4 = self.rcm4(afm4, x4)  # b 64 11 11


        # feadback
        guidance =  d1_1
        guidance1 = F.interpolate(guidance, scale_factor=1 / 8, mode='bilinear')
        guidance2 = F.interpolate(guidance, scale_factor=1 / 4, mode='bilinear')
        guidance3 = F.interpolate(guidance, scale_factor=1 / 2, mode='bilinear')
        x4 = x4 + guidance1
        x3 = x3 + guidance2
        x2 = x2 + guidance3
        x1 = x1 + guidance

        #decoder 2
        x4_1 = x4
        x3_1 = self.afm5(self.upsample(x4) , x3)
        x2_1 = self.afm5(self.upsample(x3) , x2)
        x1_1 = self.afm5(self.upsample(x2) , x1)

        x4_1 = self.decoder8(x4_1)
        x3_2 = torch.cat((x3_1, x4_1), 1)
        x3_2 = self.decoder7(x3_2)
        x2_2 = torch.cat((x2_1, x3_2), 1)
        x2_2 = self.decoder6(x2_2)
        x1_2 = torch.cat((x1_1, x2_2), 1)
        x1_2 = self.decoder5(x1_2)

        pred1 = self.unetout1(d1_1)
        pred2 = self.unetout2(x1_2)

        pred2 = F.interpolate(pred2, scale_factor=4, mode='bilinear')
        pred1 = F.interpolate(pred1, scale_factor=4, mode='bilinear')

        return pred1, pred2


if __name__ == '__main__':
    model = BiDFNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    pred2, pred1 = model(input_tensor)
    print(pred2.size())
    print(pred1.size())
