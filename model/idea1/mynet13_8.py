


import torch
import torch.nn as nn
import torch.nn.functional as F
from model import pvt_v2_b2

from model import RefineUNet

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,userelu=False):
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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels , 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out



class _DAHead(nn.Module):
    def __init__(self, in_channels):
        super(_DAHead, self).__init__()
        inter_channels = in_channels // 8
        self.conv_p1 = BasicConv2d(in_channels,inter_channels,1)
        self.conv_c1 = BasicConv2d(in_channels,inter_channels,1)
        self.pam = _PositionAttentionModule(inter_channels)
        self.cam = _ChannelAttentionModule()
        self.out= BasicConv2d(inter_channels,in_channels,1)

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_fusion = feat_p + feat_c
        feat_fusion =self.out(feat_fusion)

        return feat_fusion

class BCA(nn.Module):
    def   __init__(self, xin_channels, yin_channels, mid_channels, BatchNorm=nn.BatchNorm2d, scale=False):
        super(BCA, self).__init__()
        self.mid_channels = mid_channels
        self.f_self = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels=xin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels=yin_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(mid_channels),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=xin_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm(xin_channels),
        )
        self.scale = scale
        nn.init.constant_(self.f_up[1].weight, 0)
        nn.init.constant_(self.f_up[1].bias, 0)
        self.down = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor= 2, mode='bilinear', align_corners=True)



    def forward(self, x, y):
        xor = x
        x = self.down(x)
        y = self.down(y)
        batch_size = x.size(0)
        fself = self.f_self(x).view(batch_size, self.mid_channels, -1)
        fself = fself.permute(0, 2, 1)
        fx = self.f_x(x).view(batch_size, self.mid_channels, -1)
        fx = fx.permute(0, 2, 1)
        fy = self.f_y(y).view(batch_size, self.mid_channels, -1)
        sim_map = torch.matmul(fx, fy)
        if self.scale:
            sim_map = (self.mid_channels ** -.5) * sim_map
        sim_map_div_C = F.softmax(sim_map, dim=-1)
        fout = torch.matmul(sim_map_div_C, fself)
        fout = fout.permute(0, 2, 1).contiguous()
        fout = fout.view(batch_size, self.mid_channels, *x.size()[2:])
        out = self.f_up(fout)
        return xor + self.up(out)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,down = True):
        super(DecoderBlock, self).__init__()
        self.conv1 = BasicConv2d(in_channels, in_channels,1 )
        self.conv2 = BasicConv2d(in_channels, out_channels,1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.down = down
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.down:
          xout = self.upsample(x)
        return xout,x
class EncooderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,down = True):
        super(EncooderBlock, self).__init__()
        self.conv1 = BasicConv2d(in_channels, in_channels ,1)
        self.conv2 = BasicConv2d(in_channels, out_channels,1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class att_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(att_module, self).__init__()
        self.att_conv = nn.Sequential(
            BasicConv2d(in_channels, in_channels,1),
            BasicConv2d(in_channels, in_channels,1),
            nn.Sigmoid()
        )
        self.conv = BasicConv2d(in_channels,out_channels,1)


    def forward(self, x):
        y=x
        att_mask = self.att_conv(y)
        y = att_mask * y
        y = self.conv(y)
        return y

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



class CasAtt(nn.Module):
    def __init__(self, channel=32):
        super(CasAtt, self).__init__()
        self.conv = nn.Sequential(
                    BasicConv2d(channel,channel,1),
                     BasicConv2d(channel,channel,1),
                     nn.Sigmoid()
         )
        self.convout =BasicConv2d(channel,channel,1)

    def forward(self,x,y):
        y = self.conv(y)
        x = x*y + y
        return  self.convout(x)



class MyNet(nn.Module):
    def __init__(self, channel=32):
        super(MyNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'F:\pretrain\pvt_v2_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer1= BasicConv2d(64, channel, 1)
        self.Translayer2 = BasicConv2d(128, channel, 1)
        self.Translayer3 = BasicConv2d(320, channel, 1)
        self.Translayer4 = BasicConv2d(512, channel, 1)
        self.Translayer5 = BasicConv2d(64, channel, 1)
        self.Translayer6 = BasicConv2d(128, channel, 1)
        self.Translayer7 = BasicConv2d(320, channel, 1)
        self.Translayer8 = BasicConv2d(512, channel, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.Upsample(scale_factor=1/4, mode='bilinear', align_corners=True)

        self.down01 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.down02 = nn.Upsample(scale_factor=1/4, mode='bilinear', align_corners=True)
        self.down03 = nn.Upsample(scale_factor=1/8, mode='bilinear', align_corners=True)
        self.down04 = nn.Upsample(scale_factor=1/16, mode='bilinear', align_corners=True)

        self.upsample1 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.out1_1 =  BasicConv2d(channel*2, channel, 1)
        self.out1_2 =  BasicConv2d(channel*2, channel, 1)
        self.out1_3 =   BasicConv2d(channel*2, channel, 1)
        self.out1_4 =   BasicConv2d(channel, channel, 1)


        self.refineconv =  BasicConv2d(3, 1, 1)
        self.refine = RefineUNet(1,1)

        self.ca1 = ChannelAttention(channel)
        self.sa4= SpatialAttention()
        self.outatte = nn.Conv2d(channel, channel, 1)

       # self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=channel, out_channels=channel)
        self.decoder3 = DecoderBlock(in_channels=channel*2, out_channels=channel)
        self.decoder2 = DecoderBlock(in_channels=channel*2, out_channels=channel)
        self.decoder1=  nn.Sequential(BasicConv2d(channel*2, channel,1),
                                 BasicConv2d(channel, channel,1))

        self.decoder5= DecoderBlock(in_channels=channel*2, out_channels=channel)
        self.decoder6= DecoderBlock(in_channels=channel*2, out_channels=channel)
        self.decoder7= DecoderBlock(in_channels=channel*2, out_channels=channel)
        self.decoder8 = nn.Sequential(BasicConv2d(channel * 2, channel, 1),
                                      BasicConv2d(channel, channel, 1))

        self.encoder2 = EncooderBlock(in_channels=channel*2,out_channels=channel)
        self.encoder3 = EncooderBlock(in_channels=channel*2,out_channels=channel)
        self.encoder4 = EncooderBlock(in_channels=channel*2,out_channels=channel)






        self.unetout1 =  nn.Conv2d(channel, 1, 1)
        self.unetout2 =  nn.Conv2d(channel, 1, 1)
        self.detailout =  nn.Conv2d(channel*2, 1, 1)


        self.cobv1 =BasicConv2d(3*channel,channel,1)
        self.cobv2 =BasicConv2d(3*channel,channel,1)
        self.selayer = SELayer(channel)
        self.upconv1 =BasicConv2d(channel,channel,1)

        self.noncal = BCA(channel, channel, 16)
        self.duatt =_DAHead(channel)
        self.ca = ChannelAttention(channel*2)
        self.sa = SpatialAttention()

        self.conv = nn.Sequential(
            BasicConv2d(channel*3,channel*2,1),
            BasicConv2d(channel*2,channel*2,1),
            BasicConv2d(channel*2,channel,1)
        )
        self.edgeconv =BasicConv2d(channel,channel,1)
        self.downconv =BasicConv2d(channel*2,channel,1)
        self.catt1 =  CasAtt(channel)
        self.catt2 =  CasAtt(channel)
        self.att1 = att_module(in_channels=channel*2,out_channels=channel)
        self.att2 = att_module(in_channels=channel*2,out_channels=channel)
        self.att3 = att_module(in_channels=channel*2,out_channels=channel)
        self.att4 = att_module(in_channels=channel*2,out_channels=channel)
        self.att5 = att_module(in_channels=channel*2,out_channels=channel)
        self.att6 = att_module(in_channels=channel*2,out_channels=channel)
        self.att7 = att_module(in_channels=channel*2,out_channels=channel)
        self.att8 = att_module(in_channels=channel*2,out_channels=channel)





    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]   # 1 64 88 88
        x2 = pvt[1]   # 1 128 44 44
        x3 = pvt[2]   # 1 320 22 22
        x4 = pvt[3]   # 1 512 11 11
        xu1 = self.Translayer1(x1)
        xu2 = self.Translayer2(x2)
        xu3 = self.Translayer3(x3)
        xu4 = self.Translayer4(x4)
        xe1 = self.Translayer5(x1)
        xe2 = self.Translayer6(x2)
        xe3 = self.Translayer7(x3) # 22
        xe4 = self.Translayer8(x4)



        d1_4 ,xd2_4= self.decoder4(xu4)  # b 320 22 22

        u3 = torch.cat((d1_4, xu3), dim=1)
        d1_3 ,xd2_3= self.decoder3(u3)  # b 128 44 4
        u2 = torch.cat((d1_3, xu2), dim=1)
        d1_2 ,xd2_2= self.decoder2(u2)  # b 128 88 88
        u1 = torch.cat((d1_2, xu1), dim=1)
        d1_1 = self.decoder1(u1)



        e1 =  self.catt1(xe1,d1_1)
        e1 =self.down01(e1) # 44 44

        e2 = torch.cat((xe2,e1),dim=1)
        e2 = self.att2(e2) ## 22 22
        e2 = self.down01(e2)


        e3 = torch.cat((xe3, e2), dim=1)
        e3 = self.att3(e3) # 11 11
        e3 = self.down01(e3)

        e4 = torch.cat((xe4, e3), dim=1)
        e4 = self.att4(e4)#11


        d4 = torch.cat((e4,xd2_4),dim=1)
        d4 = self.att5(d4)
        d4 =self.upsample(d4)


        d3 = torch.cat((d4,xd2_3),dim=1)
        d3 =self.att6(d3)
        d3 =self.upsample(d3)

        d2 = torch.cat((d3, xd2_2),dim=1)
        d2 = self.att7(d2)
        d2 =self.upsample(d2)

        d1 = torch.cat((d2, d1_1),dim=1)
        d1 = self.decoder8(d1)




        pred1 = self.unetout1(d1_1)  #b 64 176 176
        pred2 = self.unetout2(d1)  #b 64 176 176
        pred2 = F.interpolate(pred2,scale_factor=4,mode='bilinear')
        pred1 = F.interpolate(pred1, scale_factor=4, mode='bilinear')
       # detailout = F.interpolate(detailout, scale_factor=4, mode='bilinear')

        return pred1,pred2


if __name__ == '__main__':
    model = MyNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    pred2,pred1= model(input_tensor)
    print(pred2.size())
    print(pred1.size())
 #   print(edgeout.size())
    # print(prediction1.size())
    # print(prediction2.size())
    # print(prediction3.size())
    # print(prediction4.size())

    # net =BCA(64,64,64)
    # a =torch.rand(1,64,44,44)
    # b =torch.rand(1,64,44,44)
    # print(net(a,b).size())