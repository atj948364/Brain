import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """注意力模块，通过门控机制增强对重要区域的关注"""

    def __init__(self, F_g, F_l, F_int):
        """
        参数:
            F_g: 来自解码器的特征图通道数
            F_l: 来自编码器的跳跃连接特征图通道数
            F_int: 中间特征图通道数
        """
        super(AttentionBlock, self).__init__()
        # 解码器特征变换
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        # 编码器特征变换
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        # 注意力权重生成
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        参数:
            g: 解码器特征
            x: 编码器跳跃连接特征
        返回:
            注意力加权后的特征
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)  # 特征融合
        psi = self.psi(psi)  # 生成注意力权重
        return x * psi  # 应用注意力权重


class AttentionUNet(nn.Module):
    """基于注意力机制的U-Net模型"""

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        """
        参数:
            in_channels: 输入图像通道数
            out_channels: 输出分割图通道数
            features: 各层特征图通道数列表
        """
        super(AttentionUNet, self).__init__()

        # 编码器（下采样）
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 构建编码器各层
        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = feature

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
        )

        # 解码器（上采样 + 注意力模块）
        self.decoder = nn.ModuleList()
        self.attention = nn.ModuleList()

        # 构建解码器各层
        for feature in reversed(features):
            # 上采样层
            self.decoder.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            # 注意力模块
            self.attention.append(
                AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2)
            )
            # 特征融合与处理
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                )
            )

        # 输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播过程
        参数:
            x: 输入图像 [batch_size, channels, height, width]
        返回:
            分割结果 [batch_size, out_channels, height, width]
        """
        skip_connections = []

        # 编码器路径
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)  # 保存跳跃连接
            x = self.pool(x)  # 下采样

        # 瓶颈层
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # 反转跳跃连接列表

        # 解码器路径（含注意力机制）
        for i in range(0, len(self.decoder), 2):
            # 上采样
            x = self.decoder[i](x)
            # 获取对应的跳跃连接
            skip_connection = skip_connections[i // 2]
            # 应用注意力机制
            attention_weights = self.attention[i // 2](g=x, x=skip_connection)
            # 拼接特征
            x = torch.cat([x, attention_weights], dim=1)
            # 卷积处理
            x = self.decoder[i + 1](x)

        # 输出层
        x = self.final_conv(x)
        return self.sigmoid(x)