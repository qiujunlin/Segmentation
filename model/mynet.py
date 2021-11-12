


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.Res2Net import  res2net50_v1b_26w_4s



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

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
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

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
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




class MyNet(nn.Module):
    def __init__(self, channel=32, n_class=1):
        super(MyNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)

        self.Combine = COM(32)
        self.out_conv = nn.Conv2d(channel, 1, 1)


        # ---- edge branch ----
        self.edge_conv1 = BasicConv2d(256, channel, kernel_size=1)
        self.edge_conv2 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.edge_conv3 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.edge_conv4 = BasicConv2d(channel, n_class, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        # ---- low-level features ----
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88

        # ---- high-level features ----
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        # ---- edge guidance ----
        x = self.edge_conv1(x1)
        x = self.edge_conv2(x)
        edge_guidance = self.edge_conv3(x)  # torch.Size([1, 64, 88, 88])


        last = self.Combine(x4_rfb,x3_rfb,x2_rfb,edge_guidance)

        last_map =  self.out_conv(last)


        lateral_edge = self.edge_conv4(edge_guidance)   # NOTES: Sup-2 (bs, 1, 88, 88) -> (bs, 1, 352, 352)
        lateral_edge = F.interpolate(lateral_edge,
                                     scale_factor=4,
                                     mode='bilinear')




        return last_map ,lateral_edge

if __name__ == '__main__':

    net  =MyNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

   # out =net(input_tensor)
    from torchsummary import summary
    summary(net,input_size=(3,352,352))
   # print(out)

