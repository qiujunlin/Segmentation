


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.Res2Net import  res2net50_v1b_26w_4s
from model.backbone.pvtv2 import pvt_v2_b2

from model.refineUnet import RefineUNet

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # CIM
        x = self.ca(x) * x  # channel attention
        attention = self.sa(x) * x  # spatial attention

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

   #     x = self.relu(x_cat + x_cat*attention)
        x = self.relu(x_cat + self.conv_res(attention)*x_cat)
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


class MyNet(nn.Module):
    def __init__(self, channel=64):
        super(MyNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'F:\pretrain\pvt_v2_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        # self.CFM = CFM(channel)
        #
        # self.SAM = SAM()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)


        self.rf1 =RF(512+320+128,channel)
        self.rf2 =RF(512+320+128,channel)
        self.rf3 =RF(512+320+128,channel)

        self.att1 = BCA(channel,channel,channel)
        self.att2 =  BCA(channel,channel,channel)
        self.att3 = BCA(channel,channel,channel)

        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

        self.down01 = nn.Upsample(scale_factor=1/8, mode='bilinear', align_corners=True)
        self.down02 = nn.Upsample(scale_factor=1/4, mode='bilinear', align_corners=True)
        self.down03 = nn.Upsample(scale_factor=1/2, mode='bilinear', align_corners=True)

        self.upsample1 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.out1 =  BasicConv2d(channel, 1, 1)
        self.out2 =  BasicConv2d(channel, 1, 1)
        self.out3 =   BasicConv2d(channel, 1, 1)
        self.out4 =   BasicConv2d(channel, 1, 1)
        self.refineconv =  BasicConv2d(3, 1, 1)
        self.refine = RefineUNet(1,1)

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()
        self.outatte = nn.Conv2d(channel, channel, 1)

        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)


        # ---- edge branch ----
        self.edge_conv1 = BasicConv2d(64, channel, kernel_size=1)
        self.edge_conv2 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.edge_conv3 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.edge_conv4 = BasicConv2d(channel, 1, kernel_size=3, padding=1)


    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]   # 1 64 88 88
        x2 = pvt[1]   # 1 128 44 44
        x3 = pvt[2]   # 1 320 22 22
        x4 = pvt[3]   # 1 512 11 11





        flusion1 = self.rf1(torch.cat((x4,self.down02(x2),self.down03(x3)),dim=1))  # 1 64 11 11
        flusion2 = self.rf2(torch.cat((self.upsample(x4),x3,self.down03(x2)),dim=1)) # # 1 64 22 2
        flusion3 = self.rf3(torch.cat((self.upsample(self.upsample(x4)),self.upsample(x3),x2),dim=1))  # 1 64 44 44




        # # CIM
        # x1 = self.ca(x1) * x1  # channel attention
        # low_feature = self.sa(x1) * x1  # spatial attention
        # low_feature =self.outatte(low_feature)’
        # ---- edge guidance ----
        x = self.edge_conv1(x1)
        x = self.edge_conv2(x)
        low_feature = self.edge_conv3(x)  # torch.Size([1, 64, 88, 88])




        att_out1 = self.att1(flusion1,self.down01(low_feature))
        att_out2 = self.att2(flusion2,self.down02(low_feature))
        att_out3 = self.att3(flusion3,self.down03(low_feature))



        out1 =self.out1(att_out1)
        out2 =self.out2(att_out2)
        out3 =self.out3(att_out3)



        prediction1 = F.interpolate(out1, scale_factor=32, mode='bilinear')
        prediction2 = F.interpolate(out2, scale_factor=16, mode='bilinear')
        prediction3 = F.interpolate(out3, scale_factor=8, mode='bilinear')


        lateral_edge = self.edge_conv4(low_feature)   # NOTES: Sup-2 (bs, 1, 88, 88) -> (bs, 1, 352, 352)
        lateral_edge = F.interpolate(lateral_edge,
                                     scale_factor=4,
                                     mode='bilinear')




        redfine1,redfine2,redfine3 = self.refine((prediction3+prediction2+prediction1))


        return prediction1,prediction2,prediction3,redfine1,redfine2,redfine3,lateral_edge



if __name__ == '__main__':
    model = MyNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2, prediction3, redfine1, redfine2, redfine3,la = model(input_tensor)
    print(prediction1.size())
    print(prediction2.size())
    print(prediction3.size())
    print(redfine1.size())
    print(redfine2.size())
    print(redfine3.size())
    print(la.size())
    # net =BCA(64,64,64)
    # a =torch.rand(1,64,44,44)
    # b =torch.rand(1,64,44,44)
    # print(net(a,b).size())