import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SwinUNet, self).__init__()
        self.model = SwinUNETR(
            # img_size=256,
            in_channels=in_channels,
            out_channels=out_channels,
            # feature_size=48,
            spatial_dims=2,  # 输入数据的空间维度。spatial_dims = 2 表示输入数据是 2D 图像
            use_checkpoint=True
        )
        # self.model = SwinUNETR(
        #     in_channels=in_channels,#输入图像的通道数
        #     out_channels=out_channels,#输出特征图的通道数
        #     patch_size=2,#将输入图像分割成的小块（patch）的大小在 Swin Transformer 架构中，图像首先会被分割成多个小块，每个小块作为一个基本处理单元。这里 patch_size = 2 表示每个小块的边长为 2 个像素（对于 2D 图像）。
        #     depths=(2, 2, 2, 2),#表示 Swin Transformer 中不同阶段（stage）的块（block）数量。
        #     num_heads=(3, 6, 12, 24),#一个序列（如 (3, 6, 12, 24)），表示 Swin Transformer 中不同阶段的多头注意力机制的头数。
        #     window_size=7,#Swin Transformer 中窗口的大小。窗口是用于计算注意力的局部区域，window_size = 7 表示窗口的边长为 7 个像素（对于 2D 图像）。
        #     qkv_bias=True,#一个布尔值，指示在生成查询（query）、键（key）和值（value）时是否使用偏置项
        #     mlp_ratio=4.0,#多层感知机（MLP）中隐藏层的维度与输入维度的比例。mlp_ratio = 4.0 表示 MLP 隐藏层的维度是输入维度的 4 倍。
        #     feature_size=48,  # 型中特征图的基本维度大小。它会影响模型中各个层的特征通道数,根据需要调整
        #     # norm_name="instance",#归一化层的名称。norm_name = "instance" 表示使用实例归一化（Instance Normalization）
        #     # drop_rate=0.0,#丢弃率，用于在全连接层中随机丢弃一部分神经元，以防止过拟合
        #     # attn_drop_rate=0.0,#注意力机制中的丢弃率，用于在多头注意力计算中随机丢弃一部分注意力权重
        #     # dropout_path_rate=0.0,#路径丢弃率，用于在 Swin Transformer 块中随机丢弃一部分路径，以提高模型的鲁棒性。
        #     # normalize=True,#一个布尔值，指示是否对输入特征进行归一化处理。normalize = True 表示进行归一化
        #     # patch_norm=False,#一个布尔值，指示是否对分块后的特征进行归一化处理
        #     use_checkpoint=True,#一个布尔值，指示是否使用检查点（checkpointing）技术
        #     spatial_dims=2,  # 输入数据的空间维度。spatial_dims = 2 表示输入数据是 2D 图像
        #     # downsample="merging",#下采样的方式。downsample = "merging" 表示使用合并操作进行下采样。
        #     use_v2=False#一个布尔值，指示是否使用 Swin Transformer 的 V2 版本
        # )

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)  # <--- 解决 BCELoss 范围问题
        # return self.model(x)
