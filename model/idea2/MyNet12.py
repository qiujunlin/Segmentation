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
class MSCA(nn.Module):
    def __init__(self, channels=32, r=4):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei


class ASPP(nn.Module):
    """
    ASPP模块
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.pyramid1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size = 1, bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True)
                                     )
        self.pyramid2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True)
                                     )
        self.pyramid3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=12, dilation=12,bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True)
                                     )
        self.pyramid4 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=18, dilation=18,bias=False),
                                      nn.BatchNorm2d(num_features=out_channels),
                                      nn.ReLU(inplace=True)
                                     )
        self.pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(num_features=out_channels),
                                     nn.ReLU(inplace=True)
                                    )
        self.output = nn.Sequential(nn.Conv2d(in_channels=5*out_channels, out_channels=out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.5)
                                   )
        self._initialize_weights()

    def forward(self, input):
        y1 = self.pyramid1(input)
        y2 = self.pyramid2(input)
        y3 = self.pyramid3(input)
        y4 = self.pyramid4(input)
        y5 = F.interpolate(self.pooling(input), size=y4.size()[2:], mode='bilinear', align_corners=True)
        out = self.output(torch.cat([y1,y2,y3,y4,y5],1))
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class Fusion(nn.Module):
    def __init__(self, inchannel, channel):
        super(Fusion, self).__init__()
        self.conv1_1 = BasicConv2d(inchannel , channel,3,1,1)
        self.conv3_1 = BasicConv2d(channel // 4, channel // 4, 3,padding=1)
        self.dconv5_1 = BasicConv2d(channel // 4, channel // 4, 3, dilation=2,padding=2)
        self.dconv7_1 = BasicConv2d(channel // 4, channel // 4, 3, dilation=3,padding=3)
        self.dconv9_1 = BasicConv2d(channel // 4, channel // 4, 3, dilation=4,padding=4)
        self.conv1_2 = BasicConv2d(channel, channel,3,1,1)
        self.conv3_3 = BasicConv2d(channel, channel, 3,1,1)

    def forward(self, x):
        x = self.conv1_1(x)
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()
        self.conv1 = BasicConv2d(in_channels, in_channels , kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels , out_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.upsample(x)

class External_attention(nn.Module):
        def __init__(self, in_c):
            super().__init__()

            self.conv1 = nn.Conv2d(in_c, in_c, 1)

            self.k = in_c * 4
            self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)

            self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)
            self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

            self.conv2 = nn.Conv2d(in_c, in_c, 1, bias=False)
            self.norm_layer = nn.GroupNorm(4, in_c)

        def forward(self, x):
            idn = x
            x = self.conv1(x)

            b, c, h, w = x.size()
            x = x.view(b, c, h * w)  # b * c * n

            attn = self.linear_0(x)  # b, k, n
            attn = F.softmax(attn, dim=-1)  # b, k, n

            attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
            x = self.linear_1(attn)  # b, c, n

            x = x.view(b, c, h, w)
            x = self.norm_layer(self.conv2(x))
            x = x + idn
            x = F.gelu(x)
            return x

class MyNet11(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64,numclass=1):
        super(MyNet11, self).__init__()
        # ---- ResNet Backbone ----,
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)




        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.down2 = nn.Upsample(scale_factor=0.25, mode='bilinear')
        self.down3 = nn.Upsample(scale_factor=0.125, mode='bilinear')
        self.down1 = nn.Upsample(scale_factor=0.5, mode='bilinear')



        self.aspp = ASPP(in_channels=2048,out_channels=64)

        self.fuse = BasicConv2d(channel*4,channel,3,1,1)
        self.out = BasicConv2d(channel,1,3,1,1)
        self.fuseatt1 = External_attention(in_c=channel)
        self.fuseatt2 = External_attention(in_c=channel)
        self.fuseatt3 = External_attention(in_c=channel)
        self.fuseatt4 = External_attention(in_c=channel)


        self.rfb1 = RFB_modified(256,out_channel=channel)
        self.rfb2 = RFB_modified(512,out_channel=channel)
        self.rfb3 = RFB_modified(1024,out_channel=channel)
        self.rfb4 = RFB_modified(2048,out_channel=channel)



        self.exatt1 = External_attention(64)
        self.exatt2 = External_attention(64)
        self.exatt3 = External_attention(64)
        self.exatt4 = External_attention(64)

        self.ca=  channel_attention(64)
        self.sa=  spatial_attention()






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

        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)
        x4 = self.rfb4(x4)

        x1_c =  x1+self.upsample(x2)+self.upsample1(x3)+self.upsample2(x4)
        x2_c = self.down1(x1)+x2+self.upsample(x3)+self.upsample1(x4)
        x3_c = self.down2(x1)+self.down1(x2)+x3+self.upsample(x4)
        x4_c = self.down3(x1)+self.down2(x2)+self.down1(x3)+x4

        fuse1 = self.fuseatt1(x1_c)
        fuse2 = self.fuseatt2(x2_c)
        fuse3 = self.fuseatt3(x3_c)
        fuse4 = self.fuseatt4(x4_c)


        fuse = torch.cat((fuse1,self.upsample(fuse2),self.upsample1(fuse3),self.upsample2(fuse4)),dim=1)

        fuse =self.fuse(fuse)
        fuse =self.ca(fuse)
        fuse =self.sa(fuse)
        fuse =self.out(fuse)


        return  self.upsample1(fuse)










if __name__ == '__main__':
   # ras = PraNet().cuda()
    from torchsummary import summary

    model = MyNet11().cuda()
    # print(torch.cuda.is_available() )
    input_tensor = torch.randn(4, 3, 352, 352).cuda()
    # down1 = nn.Upsample(scale_factor=0.25, mode='bilinear')
    # print(down1(input_tensor).size())
    a= model(input_tensor)
    print(a.size())
    # # print(b.size())
   # a= model(input_tensor)
   # print(a.size())
    # print(b.size())
    # print(c.size())
    # print(d.size())
    # print(e.size())
    # print(f.size())
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