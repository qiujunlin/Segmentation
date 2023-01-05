import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.Res2Net import res2net50_v1b_26w_4s


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
class BCA(nn.Module):
    def __init__(self, xin_channels, yin_channels, mid_channels, BatchNorm=nn.BatchNorm2d, scale=False):
        super(BCA, self).__init__()
        self.conv1 = BasicConv2d(xin_channels,yin_channels,kernel_size=3,stride=1,padding=1)
        self.conv2 = BasicConv2d(xin_channels,xin_channels,kernel_size=3,stride=1,padding=1)

        self.ca = channel_attention(xin_channels)
        self.sa = spatial_attention()


    def forward(self, x, y):
        z = torch.sigmoid(y) * x
       # z =self.conv1(z)
        z =self.conv2(z)
        z =self.ca(z)
        z =self.sa(z)


        return z

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

        self.conv1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class FinaloutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(FinaloutBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding,relu=True)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

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

class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x


# Boundary-guided fusion module
class BFM(nn.Module):
    def __init__(self, in_c, out_c, groups=8):
        super(BFM, self).__init__()

        self.mask_conv =BasicConv2d(in_c*2,out_c,kernel_size=3,stride=1,padding=1)
        self.mask_pred_conv = BasicConv2d(in_c,1,kernel_size=3,stride=1,padding=1)
        self.conv_contact = BasicConv2d(in_c*2,out_c,kernel_size=3,stride=1,padding=1)
        self.ca = channel_attention(in_c)
        self.sa = spatial_attention()


    def split_and_concate(self, x1, x2):
        N, C, H, W = x1.shape
        x2 = x2.repeat(1, self.groups, 1, 1)
        x1 = x1.reshape(N, self.groups, C // self.groups, H, W)
        x2 = x2.unsqueeze(2)
        x = torch.cat([x1, x2], dim=2)
        x = x.reshape(N, -1, H, W)
        return x

    def forward(self, low, high, mask_pred):

        mask_pred = F.interpolate(mask_pred, low.shape[2:], mode='bilinear', align_corners=True)
        mask_pred = torch.sigmoid(mask_pred)
        foreground = low * mask_pred
        background = low * (1 - mask_pred)

        mask_feature = torch.cat((foreground, background), dim=1)

        mask_feature = self.mask_conv(mask_feature)
        mask_feature =self.ca(mask_feature)
        mask_feature =self.sa(mask_feature)

        mask = self.mask_pred_conv(mask_feature)
        return mask_feature, mask



class MyNet7(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64,numclass=1):
        super(MyNet7, self).__init__()
        # ---- ResNet Backbone ----,
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb1_1 = RFB_modified(256+64, channel)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)

        self.rfb1_2 = RFB_modified(256+64, channel)
        self.rfb2_2 = RFB_modified(512, channel)
        self.rfb3_2 = RFB_modified(1024, channel)
        self.rfb4 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----

        self.out1 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.out2 = BasicConv2d(channel, 1, kernel_size=3, padding=1)
        # gt分割
        self.decoder_s4=  DecoderBlock(in_channels=channel,out_channels=channel)
        self.decoder_s3=  DecoderBlock(in_channels=channel,out_channels=channel)
        self.decoder_s2=  DecoderBlock(in_channels=channel,out_channels=channel)
        self.decoder_s1=  DecoderBlock(in_channels=channel,out_channels=channel)
        # boundary分割
        self.decoder_b4 = DecoderBlock(in_channels=channel, out_channels=channel)
        self.decoder_b3 = DecoderBlock(in_channels=channel, out_channels=channel)
        self.decoder_b2 = DecoderBlock(in_channels=channel, out_channels=channel)
        self.decoder_b1 = DecoderBlock(in_channels=channel, out_channels=channel)

        self.boudout3 = SideoutBlock(in_channels=channel,out_channels=numclass)
        self.boudout4 = SideoutBlock(in_channels=channel,out_channels=numclass)
        self.boudout2 = SideoutBlock(in_channels=channel,out_channels=numclass)
        self.boudout1 = SideoutBlock(in_channels=channel,out_channels=numclass)
        self.segout3 = SideoutBlock(in_channels=channel,out_channels=numclass)
        self.segout2 = SideoutBlock(in_channels=channel,out_channels=numclass)
        self.segout1 = SideoutBlock(in_channels=channel,out_channels=numclass)
        self.segout4 = SideoutBlock(in_channels=channel,out_channels=numclass)

        self.BFM3 = BFM(in_c=channel,out_c=channel)
        self.BFM2 = BFM(in_c=channel,out_c=channel)
        self.BFM1 = BFM(in_c=channel,out_c=channel)




        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


        self.combine1_1 = BasicConv2d(channel*2,channel,kernel_size=3,stride=1,padding=1)
        self.combine2_1 = BasicConv2d(channel*2,channel,kernel_size=3,stride=1,padding=1)
        self.combine3_1 = BasicConv2d(channel*2,channel,kernel_size=3,stride=1,padding=1)
        self.combine1_2 = BasicConv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1)
        self.combine2_2 = BasicConv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1)
        self.combine3_2 = BasicConv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1)





    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11


        x1_1 = self.rfb1_1(torch.cat((x,x1),dim=1))
        x2_1 = self.rfb2_1(x2)
        x3_1 = self.rfb3_1(x3)

        x1_2 = self.rfb1_2(torch.cat((x,x1),dim=1))
        x2_2 = self.rfb2_2(x2)
        x3_2 = self.rfb3_2(x3)


        x4 = self.rfb4(x4) # bs, channel, 11, 11


      #  x4 = self.aspp(x4)

        boundary4 = self.boudout4(x4)    # bs, channel, 11, 11
        d4=  torch.sigmoid(boundary4)*x4+x4       # bs, channel, 22, 22
        segmentation4 = self.segout4(d4)    # bs, 1, 11, 11
        # 11 11
        d4 = self.decoder_s4(x4)     # bs, channel, 22, 22
        b4 = self.decoder_b4(x4)     # bs, channel, 22, 22


        b3 =self.combine3_2(torch.cat((b4,x3_2),dim=1))       # bs, 1024, 11, 11
        # boundary  attention  3
        boundary3 = self.boudout3(b3) +self.upsample(segmentation4) # bs ,1,22,22
        x3_1 = boundary3*x3_1+x3_1
        d3,segmentation3 = self.BFM3(x3_1,d4,segmentation4)
        segmentation3 = segmentation3 +self.upsample(segmentation4)
        d3=self.decoder_s3(d3)       # bs, 1024, 22, 22
        b3=self.decoder_b3(b3)       # bs, 1024, 22, 22




        b2 =self.combine2_2(torch.cat((b3,x2_2),dim=1))
        boundary2 = self.boudout2(b2)
        x2_1 =  x2_1 *torch.sigmoid( boundary2) +x2_1
        d2, segmentation2 = self.BFM2(x2_1,d3,segmentation3)
        segmentation2 = segmentation2+self.upsample(segmentation3)
        d2=self.decoder_s2(d2)
        b2=self.decoder_b2(b3)



        d1=self.combine1_1(torch.cat((d2,x1_1),dim=1))
        b1 =self.combine1_2(torch.cat(( b2,x1_2),dim=1))

        # boundary  attention  1
        boundary1 = self.boudout1(b1)
        x1_1 =  torch.sigmoid(boundary1) * x1_1 +x1_1
        d1,segmentation1 = self.BFM1(x1_1,d2,segmentation2)
        segmentation1 =self.upsample(segmentation2) +segmentation1







        return  self.upsample1(boundary1), \
                self.upsample2(boundary2),\
                self.upsample3(boundary3),\
                self.upsample4(boundary4), \
               self.upsample1( segmentation1), \
                self.upsample3(segmentation3),  self.upsample2(segmentation2) ,\
                self.upsample1(segmentation1)











if __name__ == '__main__':
   # ras = PraNet().cuda()
    from torchsummary import summary

    model = MyNet7().cuda()
    # print(torch.cuda.is_available() )
    input_tensor = torch.randn(4, 3, 352, 352).cuda()
    # # a,b= model(input_tensor)
    # # print(a.size())
    # # print(b.size())
    a,b,c,d,e,f,g,h= model(input_tensor)
    print(a.size())
    print(b.size())
    print(c.size())
    print(d.size())
    print(e.size())
    print(f.size())
    print(g.size())
    print(h.size())
    summary(model=model, input_size=(3, 352, 352), batch_size=-1, device='cuda')

    # aspp = ASPP(in_channels=512,out_channels=512)
    # out = torch.rand(2, 512, 13, 13)
    # print(aspp(out).shape)
    # from torchsummary import summary
    #
    # model = MyNet2().cuda()
    # # input_tensor = torch.randn(1, 3, 352, 352).cuda()
    # summary(model=model, input_size=(3, 352, 352), batch_size=-1, device='cuda')