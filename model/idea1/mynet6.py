


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
        if self.userelu:
           x=self.relu(x)

        return x






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


class RRB(nn.Module):

    def __init__(self, features, out_features=512):
        super(RRB, self).__init__()

        self.unify = nn.Conv2d(features, out_features, kernel_size=1, padding=0, dilation=1, bias=False)
        self.residual = nn.Sequential(BasicConv2d(out_features,out_features,1,userelu=True),BasicConv2d(out_features,out_features,1,userelu=False))

    def forward(self, feats):
        feats = self.unify(feats)
        residual = self.residual(feats)
        return feats + residual

class CAB(nn.Module):
    def __init__(self, features):
        super(CAB, self).__init__()

        self.delta_gen1 = nn.Sequential(
            BasicConv2d(features*2,features,1,userelu=True),
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )

        self.delta_gen2 = nn.Sequential(
            BasicConv2d(features*2,features,1,userelu=True),
            nn.Conv2d(features, 2, kernel_size=3, padding=1, bias=False)
        )



    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        s = 1.0
        norm = torch.tensor([[[[h / s, w / s]]]]).type_as(input).to(input.device)
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

    def bilinear_interpolate_torch_gridsample2(self, input, size, delta=0):
        out_h, out_w = size
        n, c, h, w = input.shape
        norm = torch.tensor([[[[1, 1]]]]).type_as(input).to(input.device)

        delta_clam = torch.clamp(delta.permute(0, 2, 3, 1) / norm, -1, 1)
        grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, out_h), torch.linspace(-1, 1, out_w)),
                           dim=-1).unsqueeze(0)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)

        grid = grid.detach() + delta_clam
        output = F.grid_sample(input, grid)
        return output

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)

        concat = torch.cat((low_stage, high_stage), 1)
        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample(low_stage, (h, w), delta2)

        high_stage += low_stage
        return high_stage


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



        self.out1_1 =  BasicConv2d(64, channel, 1)
        self.out1_2 =  BasicConv2d(128, channel, 1)
        self.out1_3 =   BasicConv2d(320, channel, 1)
        self.out1_4 =   BasicConv2d(512, channel, 1)


        self.out2_1 =  BasicConv2d(64, 1, 1)
        self.out2_2 =  BasicConv2d(128, 1, 1)
        self.out2_3 =   BasicConv2d(320, 1, 1)
        self.out2_4 =   BasicConv2d(512, 1, 1)

        self.refineconv =  BasicConv2d(3, 1, 1)
        self.refine = RefineUNet(1,1)

        self.outatte = nn.Conv2d(channel, channel, 1)

        # Decoder
       # self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=channel, out_channels=channel)
        self.decoder3 = DecoderBlock(in_channels=channel*3, out_channels=channel)
        self.decoder2 = DecoderBlock(in_channels=channel*3, out_channels=channel)
        self.decoder1 = DecoderBlock(in_channels=channel*3, out_channels=channel)


        # self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder2_4 = DecoderBlock(in_channels=512, out_channels=320)
        self.decoder2_3 = DecoderBlock(in_channels=320, out_channels=128)
        self.decoder2_2 = DecoderBlock(in_channels=128, out_channels=64)
        self.decoder2_1 = DecoderBlock(in_channels=64, out_channels=64)



        # adaptive selection module
        self.asm4 = ASM(channel, 512)
        self.asm3 = ASM(channel, channel*3)
        self.asm2 = ASM(channel, channel*3)
        self.asm1 = ASM(channel, channel*3)

        self.unetout1  = nn.Sequential(BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Conv2d(32, 1, 1))






        # Sideout
        self.sideout4= SideoutBlock(32, 1)
        self.sideout3 = SideoutBlock(32, 1)
        self.sideout2 = SideoutBlock(32, 1)
        self.sideout1 = SideoutBlock(32, 1)

        self.trans4 = RRB(512,64)
        self.trans3 = RRB(320,64)
        self.trans2 = RRB(128,64)
        self.trans1 = RRB(64,64)

        self.up4 = RRB(64,64)
        self.up3 = RRB(64,64)
        self.up2 = RRB(64,64)
        self.up1 = RRB(64,64)
        self.fluse3 = CAB(64)
        self.fluse2 = CAB(64)
        self.fluse1 = CAB(64)











    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]   # 1 64 88 88
        x2 = pvt[1]   # 1 128 44 44
        x3 = pvt[2]   # 1 320 22 22
        x4 = pvt[3]   # 1 512 11 11
        x1 =self.trans1(x1)
        x2 =self.trans2(x2)
        x3 =self.trans3(x3)
        x4 =self.trans4(x4)

        ful3 =self.fluse3(x3,self.up4(x4))
        ful2 =self.fluse3(x2,self.up4(ful3))
        ful1 =self.fluse3(x1,self.up4(ful2))





        # out1_1 = self.out1_1(asm1+x1)  # b 64 176 176
        # out1_2 = self.out1_2(asm2+x2)  # b 128 88 88
        # out1_3 = self.out1_3(asm3+x3)    # b  44 44
        # out1_4 = self.out1_4(x4)   # b 64 22 22

        pred1 = self.unetout1(ful1)    # b 64 176 176


       # attention1 = self.bam1(out1_1,self.down01(pred1)) # b 64 88 88
        # attention2 = self.bam2(out1_2,self.down02(pred1)) # b 128 44 44
        # attention3 = self.bam3(out1_3,self.down03(pred1)) # b 320 22 22
        # attention4 = self.bam4(out1_4,self.down04(pred1)) # b 512 11 11
        #
        #
        #
        # out2_3 = (attention3+self.decoder2_4(attention4)) #torch.Size([1, 320, 22, 22])
        # out2_2 = (attention2+self.decoder2_3(out2_3))   # b 128 44 44
        # out2_1= (attention1+self.decoder2_2(out2_2))# b 64 88 88
        #
        #
        #
        # sideout4 = self.out2_4(attention4)
        # sideout3 = self.out2_3(out2_3)
        # sideout2 = self.out2_2(out2_2)
        # sideout1 = self.out2_1(out2_1)





        # prediction1 = F.interpolate(sideout4, scale_factor=32, mode='bilinear')
        # prediction2 = F.interpolate(sideout3, scale_factor=16, mode='bilinear')
        # prediction3 = F.interpolate(sideout2, scale_factor=8, mode='bilinear')
        # prediction4 = F.interpolate(sideout1, scale_factor=4, mode='bilinear')
        pred1 = F.interpolate(pred1, scale_factor=4, mode='bilinear')


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