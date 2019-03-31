import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, res):
        super(double_conv, self).__init__()
        self.res = res
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.LeakyReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, in_ch, 3),
            nn.BatchNorm2d(in_ch)
        )

        self.conv2 = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        if self.res:
            x = self.conv2(x.add(self.conv1(x)))
        else:
            x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, res):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, res)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate=0.3, res=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, res),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate=0.3, res=False, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if not bilinear:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch, res)
        self.drop_rate = drop_rate
        self.bilinear = bilinear

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = nn.functional.interpolate(input=x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = F.dropout(x, self.drop_rate)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet_Area(nn.Module):
    def __init__(self, scale=32, n_channels=1, n_classes=1, drop_rate=0.3, res=True):
        super(UNet_Area, self).__init__()

        # Unet part for heatmap
        self.inc = inconv(n_channels, scale, res)
        self.down1 = down(scale, scale * 2, drop_rate, res)
        self.down2 = down(scale * 2, scale * 4, drop_rate, res)
        self.down3 = down(scale * 4, scale * 8, drop_rate, res)
        self.down4 = down(scale * 8, scale * 8, drop_rate, res)
        self.up1 = up(scale * 16, scale * 4, drop_rate, res)
        self.up2 = up(scale * 8, scale * 2, drop_rate, res)
        self.up3 = up(scale * 4, scale, drop_rate, res)
        self.up4 = up(scale * 2, scale, drop_rate, res)
        self.outc = outconv(scale, n_classes)
        self.inc2 = inconv(n_classes, scale, res=True)
        self.final_conv = outconv(scale, n_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1, 1)
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # initial convolution
        x1 = self.inc(x)
        # downscale
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # deconv to heatmap
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        hm = x

        # regression to area
        x = self.inc2(x)
        x = self.final_conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.out_relu(x)
        # return heatmap and real number
        return hm, torch.squeeze(x)
