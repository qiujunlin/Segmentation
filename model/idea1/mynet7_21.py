


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
        self.selayer = SELayer(all_channels)

    def forward(self, higerencoder ,encoder, decoder):
        #decoder = self.non_local(decoder)
        fuse = torch.cat([encoder, decoder,higerencoder], dim=1)
        fuse = self.selayer(fuse)
        return fuse

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
        self.att1=  CasAtt(channel)
        self.att2=  CasAtt(channel)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 =self.att1(x2,self.upsample(x1))
        x3_1 = self.att2(x3,self.upsample(x2_1))
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1



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

        self.out1_1 =  BasicConv2d(channel*3, channel, 1)
        self.out1_2 =  BasicConv2d(channel*3, channel, 1)
        self.out1_3 =   BasicConv2d(channel*3, channel, 1)
        self.out1_4 =   BasicConv2d(channel, channel, 1)


       # self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=channel, out_channels=channel)
        self.decoder3 = DecoderBlock(in_channels=channel*3, out_channels=channel)
        self.decoder2 = DecoderBlock(in_channels=channel*3, out_channels=channel)
        self.decoder1 = nn.Sequential(BasicConv2d(channel*3, channel,1),
                                 BasicConv2d(channel, channel,1))

        # adaptive Flusion module
        self.asm3 = ASM(channel, channel*3)
        self.asm2 = ASM(channel, channel*3)
        self.asm1 = ASM(channel, channel*3)

        self.unetout1 =  nn.Conv2d(channel, 1, 1)
        self.unetout2 =  nn.Conv2d(channel, 1, 1)

        self.conv_concat2 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(2* channel,  channel, 3, padding=1)  # 最大 64*4 = 256 不大

        self.selayer = SELayer(channel)

        self.cat1 =CasAtt(channel)
        self.cat2 =CasAtt(channel)
        self.cat3 =CasAtt(channel)
        self.cat4 =CasAtt(channel)
        self.CFM = CFM(channel)


    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]   # 1 64 88 88
        x2 = pvt[1]   # 1 128 44 44
        x3 = pvt[2]   # 1 320 22 22
        x4 = pvt[3]   # 1 512 11 11
        x2_t = self.Translayer2(x2)
        x3_t = self.Translayer3(x3)
        x4_t = self.Translayer4(x4)
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)











        pred1 = self.unetout1(cfm_feature)    # b 64 176 176
        # pred2 =self.unetout2(x1_2)

       # pred2 = F.interpolate(pred2,scale_factor=4,mode='bilinear')
        pred1 = F.interpolate(pred1, scale_factor=8, mode='bilinear')

        return pred1


if __name__ == '__main__':
    model = MyNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    pred1= model(input_tensor)
  #  print(pred2.size())
    print(pred1.size())
    # print(prediction1.size())
    # print(prediction2.size())
    # print(prediction3.size())
    # print(prediction4.size())

    # net =BCA(64,64,64)
    # a =torch.rand(1,64,44,44)
    # b =torch.rand(1,64,44,44)
    # print(net(a,b).size())