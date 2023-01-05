
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


class BackBone(nn.Module):
    def __init__(self, channel=32):
        super(BackBone, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'F:\pretrain\pvt_v2_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)



        self.out = nn.Conv2d(512, 1, 1)


    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]  # 1 64 88 88
        x2 = pvt[1]  # 1 128 44 44
        x3 = pvt[2]  # 1 320 22 22
        x4 = pvt[3]  # 1 512 11 11
        pred=  self.out(x4)

        pred = F.interpolate(pred, scale_factor=32, mode='bilinear')


        return pred


if __name__ == '__main__':
    model = BiDFNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    # from torchsummary import summary
    #
    model = BiDFNet().cuda()
   # input_tensor = torch.randn(1, 3, 352, 352).cuda()
    #summary(model=model, input_size=(3, 352, 352), batch_size=-1, device='cuda')
    pre =model(input_tensor)
    print(pre.size())
    # print(prediction2.size())
    # print(prediction3.size())
    # print(prediction4.size())

    # net =BCA(64,64,64)
    # a =torch.rand(1,64,44,44)
    # b =torch.rand(1,64,44,44)
    # print(net(a,b).size())