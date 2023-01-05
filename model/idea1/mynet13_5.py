


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
        self.down = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x =self.down(x)
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_fusion = feat_p + feat_c
        feat_fusion =self.out(feat_fusion)
        feat_fusion = self.up(feat_fusion)

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
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels , kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = BasicConv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
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

class ASM(nn.Module):
    def __init__(self, in_channels, all_channels):
        super(ASM, self).__init__()
        self.non_local = NonLocalBlock(in_channels)
        self.selayer = SELayer(all_channels)

    def forward(self, higerencoder ,encoder, decoder):
        #decoder = self.non_local(decoder)
        fuse = torch.cat([encoder, decoder,higerencoder], dim=1)
        fuse = self.selayer(fuse)
        return fuse


"""
Non Local Block

https://arxiv.org/abs/1711.07971
"""


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z



class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x



class COM(nn.Module):
    def __init__(self, channel):
        super(COM, self).__init__()





        # # attention
        self.attention_conv2= BasicConv2d(channel,channel,1)
        self.attention_conv3= BasicConv2d(channel,channel,1)
        self.attention_conv4= BasicConv2d(channel,channel,1)

        self.atte2 = BCA(channel,channel,channel)
        self.atte3 = BCA(channel,channel,channel)
        self.atte4 = BCA(channel,channel,channel)


        self.conv4 = BasicConv2d(channel, channel, 3, padding=1)
        self.flu1 =  nn.Sequential(
            BasicConv2d(channel + 3,channel*2,1),
            BasicConv2d(channel*2,channel,1)
        )
        self.flu2 =  nn.Sequential(
            BasicConv2d(channel*2 + 3,channel*2,1),
            BasicConv2d(channel*2,channel*2,1),
            BasicConv2d(channel*2,channel,1),
        )
        self.flu3 = nn.Sequential(
            BasicConv2d(channel * 2+3 , channel * 2, 1),
            BasicConv2d(channel * 2 , channel * 2, 1),
            BasicConv2d(channel * 2, channel, 1),
        )
        self.down = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.down01 = nn.Upsample(scale_factor=1/4, mode='bilinear', align_corners=True)
        self.down02 = nn.Upsample(scale_factor=1/8, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)


    def forward(self, x, x1,x2):
      #  down1 = self.down01(x)  # b c 88 88
        #down2 = self.down02(x)  # b c 44 44
        x =self.down01(x)

        x2 = self.upsample1(x2)
        fluh1 = torch.cat((x, x1), dim=1)
        fluh1 = self.flu1(fluh1)
        fluh2 = torch.cat((fluh1,x,x2), dim=1)
        fluh2 = self.flu2(fluh2)
        fluh3 = torch.cat((fluh2, x1,x), dim=1)

        out = self.flu3(fluh3)

        return out


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
        self.Translayerup1 = BasicConv2d(64 ,channel, 1)
        self.Translayerup2 = BasicConv2d(512, channel, 1)

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
        self.decoder1 =  nn.Sequential(BasicConv2d(channel*2, channel,1),
                                 BasicConv2d(channel, channel,1))


        # self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder2_4 = DecoderBlock(in_channels=channel, out_channels=channel)
        self.decoder2_3 = DecoderBlock(in_channels=channel*2, out_channels=channel)
        self.decoder2_2 = DecoderBlock(in_channels=channel*2, out_channels=channel)
        self.decoder2_1 = DecoderBlock(in_channels=channel*2, out_channels=channel)




        self.unetout1 =  nn.Conv2d(channel, 1, 1)
        self.unetout2 =  nn.Conv2d(channel, 1, 1)
        self.detailout =  nn.Conv2d(channel, 1, 1)
        self.com =COM(channel)


        self.cobv1 =BasicConv2d(3*channel,channel,1)
        self.cobv2 =BasicConv2d(3*channel,channel,1)
        self.nocal = NonLocalBlock(channel)
        self.selayer = SELayer(channel)
        self.upconv1 =BasicConv2d(channel,channel,1)

        self.noncal = BCA(channel, channel, 16)
        self.duatt =_DAHead(channel)
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

        self.conv = nn.Sequential(
            BasicConv2d(channel*2,channel*2,1),
            BasicConv2d(channel*2,channel,1)
        )
        self.edgeconv =BasicConv2d(channel,channel,1)
        self.downconv =BasicConv2d(channel,channel,1)




    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]   # 1 64 88 88
        x2 = pvt[1]   # 1 128 44 44
        x3 = pvt[2]   # 1 320 22 22
        x4 = pvt[3]   # 1 512 11 11
        xu1 =self.Translayer1(x1)
        xu2 =self.Translayer2(x2)
        xu3 =self.Translayer3(x3)
        xu4 =self.Translayer4(x4)
        upx1 =self.Translayerup1(x1)
        upx2 =self.Translayerup2(x4)

        #mutipul
        xu3 = self.upsample(xu4)*xu3
        xu2 = self.upsample(xu3)*xu2
        xu1 = self.upsample(xu2)*xu1
        #decoder1
        d1_4 = self.decoder4(xu4)  # b 320 22 22

        u3 =torch.cat((d1_4,xu3),dim=1)
        d1_3 = self.decoder3(u3)  # b 128 44 4
        u2 = torch.cat((d1_3,xu2),dim=1)
        d1_2 = self.decoder2(u2)  # b 128 88 88
        u1 =torch.cat((d1_2,xu1),dim=1)
        d1_1 = self.decoder1(u1)

        #flusion
        outflu = self.com(x,upx1,upx2) # h w
        # spitial and channel
        outflu = self.ca(outflu) * outflu  # channel attention
        outflu = self.sa(outflu) * outflu  # spatial attention
        #
        att1 =d1_1
        att2 = self.downconv(outflu)
        out2 = self.noncal(att1,att2)

        # out
        detailout = self.detailout(outflu)
        pred1 = self.unetout1(d1_1)  #b 64 176 176
        pred2 =self.unetout2(out2)
        pred2 = F.interpolate(pred2,scale_factor=4,mode='bilinear')
        pred1 = F.interpolate(pred1, scale_factor=4, mode='bilinear')
        detailout = F.interpolate(detailout, scale_factor=4, mode='bilinear')

        return pred1,pred2,detailout


if __name__ == '__main__':
    model = MyNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    pred2,pred1,edgeout= model(input_tensor)
    print(pred2.size())
    print(pred1.size())
    print(edgeout.size())
