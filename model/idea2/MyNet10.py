import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.Res2Net import res2net50_v1b_26w_4s
from model.backbone.pvtv2 import pvt_v2_b2

import torch.nn.init as init

class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,relu=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            channel_attention(out_channel),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3),
            channel_attention(out_channel),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5),
            channel_attention(out_channel)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7),
            channel_attention(out_channel)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)
        self.sa = spatial_attention()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        x=self.sa(x)
        return x







class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1





class Fusion(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(Fusion, self).__init__()
        self.relu = nn.ReLU(True)
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)

        self.fuseconv = BasicConv2d(channel, channel, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.conv_high = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv_low = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, high, low):
        low = self.conv1(low)
        high = self.conv2(high)

        avg_low = torch.mean(low, dim=1, keepdim=True)
        max_low, _ = torch.max(low, dim=1, keepdim=True)
        avg_high = torch.mean(high, dim=1, keepdim=True)
        max_high, _ = torch.max(high, dim=1, keepdim=True)

        avg_low_fu = avg_low * self.upsample(max_high)
        max_low_fu = max_low * self.upsample(avg_high)
        avg_high_fu = avg_high * self.downsample(max_low)
        max_high_fu = max_high * self.downsample(avg_low)

        low_fuse = self.conv_low(torch.cat((avg_low_fu, max_low_fu),dim=1))
        high_fuse = self.conv_high(torch.cat((avg_high_fu, max_high_fu), dim=1))

        low = self.sigmoid(low_fuse) * low
        high = self.sigmoid(high_fuse) * high

        fuse = self.fuseconv(low +self.upsample( high))

        return  fuse

class MyNet10(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32,numclass=1):
        super(MyNet10, self).__init__()
        # ---- ResNet Backbone ----,
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
      #  path = '/root/autodl-fs/pvt_v2_b3.pth'
        path = 'F:\pretrain/pvt_v2_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # ---- Receptive Field Block like module ----
        self.rfb1 = RFB_modified(64, channel)
        self.rfb2 = RFB_modified(128, channel)
        self.rfb3 = RFB_modified(320, channel)
        self.rfb4 = RFB_modified(512, channel)

        self.cfm =CFM(channel)

        self.fuse =Fusion(channel)


        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')

        self.out_1= nn.Conv2d(channel, 1, 1)
        self.out_2 = nn.Conv2d(channel, 1, 1)






    def forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]  # 1 64 88 88
        x2 = pvt[1]  # 1 128 44 44
        x3 = pvt[2]  # 1 320 22 22
        x4 = pvt[3]  # 1 512 11 11


        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)
        x4 = self.rfb4(x4)

        cfm_feature = self.cfm(x4, x3, x2)

        fush = self.fuse(cfm_feature,x1)

        out1 = self.out_1(cfm_feature)
        out2 = self.out_2(fush)

        prediction1 = self.upsample2(out1)
        prediction2 = self.upsample1(out2)


        return prediction1,prediction2










if __name__ == '__main__':
   # ras = PraNet().cuda()
    from torchsummary import summary

    model = MyNet10().cuda()
    # print(torch.cuda.is_available() )
    input_tensor = torch.randn(4, 3, 352, 352).cuda()
    # # a,b= model(input_tensor)
    # # print(a.size())
    # # print(b.size())
    a,b= model(input_tensor)
    print(a.size())
    print(b.size())

    summary(model=model, input_size=(3, 352, 352), batch_size=-1, device='cuda')

    # aspp = ASPP(in_channels=512,out_channels=512)
    # out = torch.rand(2, 512, 13, 13)
    # print(aspp(out).shape)
    # from torchsummary import summary
    #
    # model = MyNet2().cuda()
    # # input_tensor = torch.randn(1, 3, 352, 352).cuda()
    # summary(model=model, input_size=(3, 352, 352), batch_size=-1, device='cuda')