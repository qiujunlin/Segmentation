


import torch
import torch.nn as nn
import torch.nn.functional as F
from model import pvt_v2_b2


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



class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h



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

    def forward(self, x, y):
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
        return x + out
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = BasicConv2d(in_channels // 4, out_channels, kernel_size=kernel_size,
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
        decoder = self.non_local(decoder)
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
        self.relu = nn.ReLU(True)

        self.co1_1 = BasicConv2d(channel,channel,kernel_size=1)
        self.co1_2 = BasicConv2d(channel*2,channel,kernel_size=1)
        self.co1_3 = BasicConv2d(channel,channel,kernel_size=1)
        self.co1_4 = BasicConv2d(channel,channel,kernel_size=1)

        self.co2_1 = BasicConv2d(channel, channel, kernel_size=1)
        self.co2_2 = BasicConv2d(channel + channel, channel, kernel_size=1)
        self.co2_3 = BasicConv2d(channel, channel, kernel_size=1)
        self.co2_4 = BasicConv2d(channel, channel, kernel_size=1)

        self.co3_1 = BasicConv2d(channel, channel, kernel_size=1)
        self.co3_2 = BasicConv2d(channel + channel, channel, kernel_size=1)
        self.co3_3 = BasicConv2d(channel, channel, kernel_size=1)
        self.co3_4 = BasicConv2d(channel, channel, kernel_size=1)


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3,edge_guidance):

        """
        edge_guidance  -> x1 bs 32 88 88
        x1 -> x4   bs 32 11 11
        x2 -> x3  bs  32 22 22
        x3 -> x2  bs  32 44 44

        1. 先进行downSample 然后cat
        2. 然后卷积之后相乘


        1. edge_guidance1和x1 拼接  然后conv
        2. edge_guidance2和x2 拼接  然后conv 然和和 第一步结果相乘
        3. edge_guidance3和x3 拼接  然后conv  然后和第二部结果相乘
        """

        #x1_1 = x1  # 32,88, 88
        edge_guidance1 = F.interpolate(edge_guidance, scale_factor=1/8, mode='bilinear')
        edge_guidance2 = F.interpolate(edge_guidance, scale_factor=1/4, mode='bilinear')
        edge_guidance3 = F.interpolate(edge_guidance, scale_factor=1/2, mode='bilinear')

        x1_1 = torch.cat((self.co1_1(x1),edge_guidance1),dim=1)
        x1_2 = F.relu(self.co1_2(x1_1))
        x1_3 = F.relu(self.co1_3(x1_2))
        x1_4 = F.relu(self.co1_4(x1_3))

        x2_1 = torch.cat((self.co1_1(x2), edge_guidance2), dim=1)
        x2_2 = F.relu(self.co1_2(x2_1))
        x2_3 = F.relu(self.co1_3(x2_2))
        x2_4 = F.relu(self.co1_4(x2_3))
        x2_5 = self.upsample(x1_4)*x2_4


        x3_1 = torch.cat((self.co1_1(x3), edge_guidance3), dim=1)
        x3_2 = F.relu(self.co1_2(x3_1))
        x3_3 = F.relu(self.co1_3(x3_2))
        x3_4 = F.relu(self.co1_4(x3_3))
        x3_5 = self.upsample(x2_5)*x3_4

        x = torch.cat([self.upsample(self.upsample(x1_4)) ,self.upsample(x2_5), x3_5],dim=1)
        x = self.conv4(x)
        return  x
class SEB3(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SEB3,self).__init__()
        self.conv1=BasicConv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1)
        self.upsample1=nn.Upsample(scale_factor=2,mode="bilinear")

    def forward(self,x1,x2):
        return x1*(self.upsample1(x2))
class SEB2(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SEB2,self).__init__()
        self.conv1=BasicConv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2 = BasicConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample1=nn.Upsample(scale_factor=2,mode="bilinear")
        self.upsample2 = nn.Upsample(scale_factor=4, mode="bilinear")
    def forward(self,x1,x2,x3):
        return x1*(self.upsample1(x2))*(self.upsample2(x3))

class SEB1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(SEB1,self).__init__()
        self.conv1=BasicConv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1)
        self.conv2=BasicConv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1)
        self.conv3=BasicConv2d(in_channels,out_channels,kernel_size=3,stride=1, padding=1)
        self.upsample1=nn.Upsample(scale_factor=2,mode="bilinear")
        self.upsample2=nn.Upsample(scale_factor=4,mode="bilinear")
        self.upsample3=nn.Upsample(scale_factor=8,mode="bilinear")
    def forward(self,x1,x2,x3,x4):
        return x1*(self.upsample1(x2))*(self.upsample2(x3))*(self.upsample3(x4))

class RRB(nn.Module):
    def __init__(self, features, out_features=512):
        super(RRB, self).__init__()
        self.unify = nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False)
        self.residual = nn.Sequential(BasicConv2d(out_features,out_features,1,userelu=True),BasicConv2d(out_features,out_features,1,userelu=False))

    def forward(self, feats):
        feats = self.unify(feats)
        residual = self.residual(feats)
        return feats + residual
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


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)


        self.down01 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.down02 = nn.Upsample(scale_factor=1/4, mode='bilinear', align_corners=True)
        self.down03 = nn.Upsample(scale_factor=1/8, mode='bilinear', align_corners=True)
        self.down04 = nn.Upsample(scale_factor=1/16, mode='bilinear', align_corners=True)

        self.upsample1 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)



        # Decoder
       # self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=channel, out_channels=channel)
        self.decoder3 = DecoderBlock(in_channels=channel*2, out_channels=channel)
        self.decoder2 = DecoderBlock(in_channels=channel*2, out_channels=channel)
        self.decoder1 = DecoderBlock(in_channels=channel*2, out_channels=channel)






        # adaptive selection module
        self.asm4 = ASM(channel, 512)
        self.asm3 = ASM(channel, channel*3)
        self.asm2 = ASM(channel, channel*3)
        self.asm1 = ASM(channel, channel*3)

        self.unetout1  = nn.Sequential(BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(32, 1, 1))
        self.unetout1 = nn.Sequential(nn.Conv2d(channel, 1, 1))





        # Sideout
        self.sideout4= SideoutBlock(32, 1)
        self.sideout3 = SideoutBlock(32, 1)
        self.sideout2 = SideoutBlock(32, 1)
        self.sideout1 = SideoutBlock(32, 1)
        self.seb1 =SEB1(channel,channel)
        self.seb2 =SEB2(channel,channel)
        self.seb3 =SEB3(channel,channel)
        self.rrb4 =RRB(channel,channel)
        self.rrb3 =RRB(channel,channel)
        self.rrb2 =RRB(channel,channel)
        self.rrb1 =RRB(channel,channel)




    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]   # 1 64 88 88
        x2 = pvt[1]   # 1 128 44 44
        x3 = pvt[2]   # 1 320 22 22
        x4 = pvt[3]   # 1 512 11 11
        x1 =self.Translayer1(x1)
        x2 =self.Translayer2(x2)
        x3 =self.Translayer3(x3)
        x4 =self.Translayer4(x4)


        x1 =self.upsample(x2)*x1
        x2 =x2*self.upsample(x3)
        x3 =self.upsample(x4)*x3






        d1_4 =self.decoder4(x4) # b 320 22 22
        d1_3 =self.decoder3(torch.cat((d1_4,x3),dim=1))  # b 128 44 4
        d1_2 =self.decoder2(torch.cat((d1_3,x2),dim=1))  # b 128 88 88
        d1_1 =self.decoder1(torch.cat((d1_2,x1),dim=1))  # b 64 176 176
        pred1 = self.unetout1(d1_1)    # b 64 176 176
        pred1 = F.interpolate(pred1, scale_factor=2, mode='bilinear')


        return pred1



if __name__ == '__main__':
    model = MyNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    pred1= model(input_tensor)
    print(pred1.size())
    # print(prediction1.size())
    # print(prediction2.size())
    # print(prediction3.size())
    # print(prediction4.size())

    # net =BCA(64,64,64)
    # a =torch.rand(1,64,44,44)
    # b =torch.rand(1,64,44,44)
    # print(net(a,b).size())