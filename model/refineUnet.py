from torch import nn
import  torch

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv =nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class RefineUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(RefineUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inconv =  nn.Sequential(
            nn.Conv2d(n_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down1 = Down(64, 64)
        self.down2 = Down(64, 64)
        self.down3 = Down(64, 64)
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        self.up3 = Up(128, 64)
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_up2 = self.up1(x4, x3)
        x_up3 = self.up2(x_up2, x2)
        x_up4 = self.up3(x_up3, x1)

        logits1 = self.outconv(x_up2)
        logits2 = self.outconv(x_up3)
        logits3 = self.outconv(x_up4)
        return logits3,self.upsample1(logits2),self.upsample2(logits1)
if __name__ == '__main__':
    net = RefineUNet(1, 1)
    a= torch.rand(1, 1, 352, 352)
    o1,o2,o3 = net(a)
    print(o1.size())
    print(o2.size())
    print(o3.size())

