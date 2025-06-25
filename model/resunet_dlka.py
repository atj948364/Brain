import torch
import torch.nn as nn
import torch.nn.functional as F


class Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.second(self.first(x))


class DLKA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise_large = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=7, padding=6, groups=channels, dilation=2),
            nn.Conv2d(channels, channels, kernel_size=9, padding=12, groups=channels, dilation=3),
        )
        self.depthwise_small = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        large = self.depthwise_large(x)
        small = self.depthwise_small(x)
        out = self.pointwise(large + small)
        out = self.bn(out)
        avg_out = self.fc(self.avg_pool(out))
        max_out = self.fc(self.max_pool(out))
        channel_att = self.sigmoid(avg_out + max_out)
        return self.activation(out * channel_att + x)


class UResnet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block=BasicBlock, layers=[3, 4, 6, 3], use_dlka=False):
        super().__init__()
        nb_filter = [64, 128, 256, 512, 1024]
        self.Up = Up()
        self.use_dlka = use_dlka
        self.in_channel = nb_filter[0]
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        if self.use_dlka:
            self.dlka3 = DLKA(nb_filter[3] * block.expansion)
            self.dlka2 = DLKA(nb_filter[2] * block.expansion)
            self.dlka1 = DLKA(nb_filter[1] * block.expansion)
            self.dlka0 = DLKA(nb_filter[0])

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3], nb_filter[3] * block.expansion)
        self.conv2_2 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2], nb_filter[2] * block.expansion)
        self.conv1_3 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1], nb_filter[1] * block.expansion)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, middle_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0, x3_0)], 1))
        if self.use_dlka:
            x3_1 = self.dlka3(x3_1)

        x2_2 = self.conv2_2(torch.cat([x2_0, self.Up(x3_1, x2_0)], 1))
        if self.use_dlka:
            x2_2 = self.dlka2(x2_2)

        x1_3 = self.conv1_3(torch.cat([x1_0, self.Up(x2_2, x1_0)], 1))
        if self.use_dlka:
            x1_3 = self.dlka1(x1_3)

        x0_4 = self.conv0_4(torch.cat([x0_0, self.Up(x1_3, x0_0)], 1))
        if self.use_dlka:
            x0_4 = self.dlka0(x0_4)

        return torch.sigmoid(self.final(x0_4))
