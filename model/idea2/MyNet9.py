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
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,relu=True):
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


class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(channel, channel, 1),

        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.Conv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, channel, 3, padding=3, dilation=3),

        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.Conv2d(channel, channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(channel, channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, channel, 3, padding=5, dilation=5),

        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.Conv2d(channel, channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(channel, channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, channel, 3, padding=7, dilation=7),

        )
        self.conv_cat = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_contact = BasicConv2d(channel*4, channel, 3,padding=1)

    def forward(self, low,high):
        # y low  x
        high= F.interpolate(high, size=low.size()[2:], mode='bilinear', align_corners=False)
        x = self.conv_cat(torch.cat((low,high),dim=1))
        x0 = self.branch0(x)
        x1 = self.branch1(x+x0)
        x2 = self.branch2(x+x1)
        x3 = self.branch3(x+x2)
        x_cat = self.conv_contact(torch.cat((x0, x1, x2, x3),dim=1))
        x =x+x_cat
        return x
class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class FinaloutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(FinaloutBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding,relu=True)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels , out_channels, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x=self.upsample(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels , kernel_size=kernel_size,
                               stride=stride, padding=padding,relu=True)

        self.conv2 = BasicConv2d(in_channels   , out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding,relu=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x= self.upsample(x)
        return x




class BoundaryDecoder(nn.Module):
    def __init__(self, midchannel=256):
        super(BoundaryDecoder, self).__init__()
        self.compress1 = BasicConv2d(256,midchannel,kernel_size=3,stride=1,padding=1)
        self.compress2 = BasicConv2d(512,midchannel,kernel_size=3,stride=1,padding=1)
        self.compress3 = BasicConv2d(1024,midchannel,kernel_size=3,stride=1,padding=1)
        self.compress4 = BasicConv2d(2048,midchannel,kernel_size=3,stride=1,padding=1)
        self.locate1 =  BasicConv2d(midchannel,midchannel,kernel_size=3,stride=1,padding=1)
        self.locate2 =  BasicConv2d(midchannel,midchannel,kernel_size=3,stride=1,padding=1)
        self.locate3 =  BasicConv2d(midchannel,midchannel,kernel_size=3,stride=1,padding=1)
        self.locate4 =  BasicConv2d(midchannel,midchannel,kernel_size=3,stride=1,padding=1)
        self.predict = nn.Conv2d(midchannel, 1, 3, 1, 1)

    def upsample(self, x,size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x1,x2, x3, x4):
        x2 = self.compress2(x2)
        x3 = self.compress3(x3)
        x4 = self.compress4(x4)
        x1 = self.compress1(x1)

        x4 = self.locate1(x4)
        x3 = x3 + self.upsample(x4, x3.shape[2:])
        x3 = self.locate2(x3)
        x2 = x2 + self.upsample(x3, x2.shape[2:])
        x2 = self.locate3(x2)

        x1 = x1 + self.upsample(x2,x1.shape[2:])
        x1 = self.locate4(x1)

        edge_attention = self.predict(x1)

        return edge_attention


class BGA(nn.Module):
    def __init__(self, channel,groups=8):
        super(BGA, self).__init__()

        self.groups=groups
        sc_channel = (channel // groups + 1) * groups
        self.conv1 = BasicConv2d(sc_channel,sc_channel,3,padding=1)
        self.conv2 = BasicConv2d(sc_channel,channel,3,padding=1)

    def split_and_concate(self, x1, x2):
        N, C, H, W = x1.shape
        x2 = x2.repeat(1, self.groups, 1, 1)
        x1 = x1.reshape(N, self.groups, C // self.groups, H, W)
        x2 = x2.unsqueeze(2)
        x = torch.cat([x1, x2], dim=2)
        x = x.reshape(N, -1, H, W)
        return x
    def forward(self, x, edge):
        if x.size() != edge.size():
            edge = F.interpolate(edge, x.size()[2:], mode='bilinear', align_corners=False)
        edge_attention = torch.sigmoid(edge)
        x =  self.conv1(self.split_and_concate(x,edge_attention))
        x =  self.conv2(x)

        return x


class MyNet9(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=128,numclass=1):
        super(MyNet9, self).__init__()
        # ---- ResNet Backbone ----,
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        # ---- Receptive Field Block like module ----
        self.rfb1 = RFB_modified(256, channel)
        self.rfb2 = RFB_modified(512, channel)
        self.rfb3 = RFB_modified(1024, channel)
        self.rfb4 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----

        self.out1 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.out2 = BasicConv2d(channel, 1, kernel_size=3, padding=1)


        self.segout3 = SideoutBlock(in_channels=channel,out_channels=numclass)
        self.segout2 = SideoutBlock(in_channels=channel,out_channels=numclass)
        self.segout1 = SideoutBlock(in_channels=channel,out_channels=numclass)


        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')


        self.combine1 = Fusion(channel)
        self.combine2 = Fusion(channel)
        self.combine3 = Fusion(channel)

        self.BGA1 =BGA(channel)
        self.BGA2 =BGA(channel)
        self.BGA3 =BGA(channel)
        self.BGA4 =BGA(channel)


        self.boundary_decoder =  BoundaryDecoder(256)




    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        edge = self.boundary_decoder(x1, x2, x3, x4)

        x1 = self.rfb1(x1)
        x2=  self.rfb2(x2)
        x3 = self.rfb3(x3)
        x4 = self.rfb4(x4)

        x1 = self.BGA1(x1,edge)
        x2 = self.BGA2(x2,edge)
        x3 = self.BGA3(x3,edge)
        x4 = self.BGA4(x4,edge)


        d3 =self.combine3(x3,x4)       # bs, 1024, 11, 11

        segmentation3 = self.segout3(d3)

        d2 = self.combine2(x2,d3)
        segmentation2 = self.segout2(d2)


        d1=self.combine1(x1,d2)

        segmentation1 = self.segout1(d1)


        return    self.upsample1(edge) ,self.upsample3(segmentation3),self.upsample2(segmentation2),self.upsample1(segmentation1)










if __name__ == '__main__':
   # ras = PraNet().cuda()
    from torchsummary import summary

    model = MyNet9().cuda()
    # print(torch.cuda.is_available() )
    input_tensor = torch.randn(4, 3, 352, 352).cuda()
    # # a,b= model(input_tensor)
    # # print(a.size())
    # # print(b.size())
    a,b,c,d= model(input_tensor)
    print(a.size())
    print(b.size())
    print(c.size())
    print(d.size())
    # print(e.size())
    # print(f.size())
    # print(g.size())
    # print(h.size())
  #  summary(model=model, input_size=(3, 352, 352), batch_size=-1, device='cuda')

    # aspp = ASPP(in_channels=512,out_channels=512)
    # out = torch.rand(2, 512, 13, 13)
    # print(aspp(out).shape)
    # from torchsummary import summary
    #
    # model = MyNet2().cuda()
    # # input_tensor = torch.randn(1, 3, 352, 352).cuda()
    # summary(model=model, input_size=(3, 352, 352), batch_size=-1, device='cuda')