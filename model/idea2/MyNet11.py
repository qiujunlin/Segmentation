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
    def __init__(self, midchannel=64):
        super(BoundaryDecoder, self).__init__()
        self.compress1 = BasicConv2d(256,midchannel,kernel_size=3,stride=1,padding=1)
        self.compress2 = BasicConv2d(512,midchannel,kernel_size=3,stride=1,padding=1)
        self.compress3 = BasicConv2d(1024,midchannel,kernel_size=3,stride=1,padding=1)
        self.compress4 = BasicConv2d(2048,midchannel,kernel_size=3,stride=1,padding=1)
        self.conv_contact = BasicConv2d(midchannel*4,midchannel,kernel_size=3,stride=1,padding=1)
        self.predict = nn.Conv2d(midchannel, 1, 3, 1, 1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def upsample(self, x,size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x1,x2, x3, x4):
        x2 = self.compress2(x2)
        x3 = self.compress3(x3)
        x4 = self.compress4(x4)
        x1 = self.compress1(x1)

        x2 =self.upsample2(x2)
        x3 =self.upsample3(x3)
        x4 =self.upsample4(x4)


        edge_attention = self.conv_contact(torch.cat((x1,x2,x3,x4),dim=1))

        edge_attention =self.predict(edge_attention)

        return edge_attention


class BGA(nn.Module):
    def __init__(self, channel,groups=8):
        super(BGA, self).__init__()

        self.groups=groups
        sc_channel = (channel // groups + 1) * groups
        self.convforground= BasicConv2d(sc_channel,sc_channel,3,padding=1)
        self.convboundary = BasicConv2d(sc_channel,sc_channel,3,padding=1)
        self.convbackground = BasicConv2d(sc_channel,sc_channel,3,padding=1)
        self.convcontact = BasicConv2d(sc_channel*3,channel,3,padding=1)

    def split_and_concate(self, x1, x2):
        N, C, H, W = x1.shape
        x2 = x2.repeat(1, self.groups, 1, 1)
        x1 = x1.reshape(N, self.groups, C // self.groups, H, W)
        x2 = x2.unsqueeze(2)
        x = torch.cat([x1, x2], dim=2)
        x = x.reshape(N, -1, H, W)
        return x
    def forward(self, x, edge,mask):
        residual =x
        if x.size() != edge.size():
            edge = F.interpolate(edge, x.size()[2:], mode='bilinear', align_corners=False)
        if x.size() != mask.size():
            mask = F.interpolate(mask, x.size()[2:], mode='bilinear', align_corners=False)
        edge_attention = torch.sigmoid(edge)
        mask_attention = torch.sigmoid(mask)

        # boundary
        boundary_att  = torch.abs(mask_attention - 0.5)
        boundary_att = 1 - (boundary_att  / 0.5)
        boundary_x = x * boundary_att

        # foregound
        foregound_att = mask_attention
        foregound_att = torch.clip(foregound_att - boundary_att, 0, 1)
        foregound_x = x * foregound_att

        # background
        background_att = 1 - mask_attention
        background_att = torch.clip(background_att - boundary_att, 0, 1)
        background_x = x * background_att



        boundary_x =    self.convboundary(self.split_and_concate(boundary_x,edge_attention))
        background_x =    self.convbackground(self.split_and_concate(background_x,edge_attention))
        foregound_x =    self.convforground(self.split_and_concate(foregound_x,edge_attention))

        mask_feature = torch.cat((boundary_x, background_x,foregound_x), dim=1)

        out = self.convcontact(mask_feature) + residual


        return out


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        agg = self.conv4(x3_2)
        edge = self.conv5(agg)

        return agg,edge


class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample = BasicConv2d(channel, channel, 3, padding=1)
        self.cat_conv = BasicConv2d(channel*2, channel, 3, padding=1)
    def forward(self, x_low, x_high):
        x_cat = torch.cat((x_low, x_high), dim=1)
        x_cat = self.cat_conv(x_cat)

        return x_cat

class MyNet11(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32,numclass=1):
        super(MyNet11, self).__init__()
        # ---- ResNet Backbone ----,
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        # ---- Receptive Field Block like module ----
        self.rfb1 = RFB_modified(256, channel)
        self.rfb2 = RFB_modified(512, channel)
        self.rfb3 = RFB_modified(1024, channel)
        self.rfb4 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----

        self.segout4 = BasicConv2d(channel,1,3,padding=1)
        self.segout3 = BasicConv2d(channel,1,3,padding=1)
        self.segout2 = BasicConv2d(channel,1,3,padding=1)


        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


        self.combine1 = Fusion(channel)
        self.combine2 = Fusion(channel)
        self.combine3 = Fusion(channel)


        self.BGA2 =BGA(channel)
        self.BGA3 =BGA(channel)
        self.BGA4 =BGA(channel)

        self.fuse4 = Fusion(channel=channel)
        self.fuse3 = Fusion(channel=channel)
        self.fuse2 = Fusion(channel=channel)


        self.boundary_decoder =  BoundaryDecoder(256)

        self.agg =aggregation(channel)




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

        agg,out5 =self.agg(x4,x3,x2)  # (bs, 1, 44, 44)

        predict5 = F.interpolate(out5, scale_factor=8,
                                      mode='bilinear')  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)


        out5  =  F.interpolate(out5, scale_factor=0.25,mode='bilinear')
        x4 = self.BGA4(x4,edge,out5)  # 需要输出 x4

        x4 = self.fuse4(x4,self.upsample4(agg))


        out4 = self.segout4(x4)

        out4 =  out4 +out5

        predict4 = F.interpolate(out4, scale_factor=32,
                                      mode='bilinear')

        x3 = self.BGA3(x3,edge,out4)

        out4 =  F.interpolate(out4, scale_factor=2,mode='bilinear')

        x3 = self.fuse3(x3, self.upsample(x4))
        out3 = self.segout3(x3)
        out3 = out3 + out4

        predict3 = F.interpolate(out3, scale_factor=16,mode='bilinear')

        x2 = self.BGA2(x2, edge, out3)


        out3 = F.interpolate(out3, scale_factor=2,mode='bilinear')

        x2 = self.fuse2(x2,self.upsample(x3))
        out2 = self.segout2(x2)
        out2 = out2 + out3

        predict2 = F.interpolate(out2, scale_factor=8,
                                 mode='bilinear')

        edge =self.upsample1(edge)



        return  predict2,predict3,predict4,predict5,edge










if __name__ == '__main__':
   # ras = PraNet().cuda()
    from torchsummary import summary

    model = MyNet11().cuda()
    # print(torch.cuda.is_available() )
    input_tensor = torch.randn(4, 3, 352, 352).cuda()
    # # a,b= model(input_tensor)
    # # print(a.size())
    # # print(b.size())
    a,b,c,d,e= model(input_tensor)
    print(a.size())
    print(b.size())
    print(c.size())
    print(d.size())
    print(e.size())
    # print(f.size())
    # print(g.size())
    # print(h.size())
    summary(model=model, input_size=(3, 352, 352), batch_size=-1, device='cuda')

    # aspp = ASPP(in_channels=512,out_channels=512)
    # out = torch.rand(2, 512, 13, 13)
    # print(aspp(out).shape)
    # from torchsummary import summary
    #
    # model = MyNet2().cuda()
    # # input_tensor = torch.randn(1, 3, 352, 352).cuda()
    # summary(model=model, input_size=(3, 352, 352), batch_size=-1, device='cuda')