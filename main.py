from omegaconf import OmegaConf
import os
import torch
from dataloader import prepare_dataloaders
from model import UNet_with_Residual, AttentionUNet, SwinUNet
from train import train_and_evaluate,print_model_summary  # 新增导入 print_model_summary


def main():
    config = OmegaConf.load(os.path.join("config", "config.yaml"))

    DATASET_DIR = config.paths.data.dataset_dir# 获取配置中的路径信息
    weights_path = config.paths.weights_root
    plots_path = config.paths.plots_root
    saves_path = config.paths.saves_root

    os.makedirs(DATASET_DIR, exist_ok=True)# 创建必要的目录
    os.makedirs(weights_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(saves_path, exist_ok=True)

    device = torch.device(config.training.device)# 设置设备（GPU 或 CPU）

    dataloaders, dataset_sizes = prepare_dataloaders(  # 准备数据加载器
        gdrive_url=config.data.gdrive_url,
        dataset_zip_path=config.paths.data.zip_path,
        dataset_path=config.paths.data.dataset_dir,
        save_path=config.paths.saves_root,
        batch_size=config.training.batch_size,
        augmentation=config.training.augmentation,
        train_ratio=config.data.train_ratio,
        valid_ratio=config.data.valid_ratio,
        seed=config.seed,
    )

    train_size = dataset_sizes["train"]
    valid_size = dataset_sizes["val"]
    test_size = dataset_sizes["test"]

    print(f"训练集样本数量: {train_size}")
    print(f"验证集样本数量: {valid_size}")
    print(f"测试集样本数量: {test_size}")
    print(f"模型名称: {config.training.model}")

    if config.training.model == "UNet_with_Residual": # 根据配置选择模型
        model = UNet_with_Residual()
    elif config.training.model == "AttentionUNet": # 根据配置选择模型
        model = AttentionUNet()
    elif config.training.model == "SwinUNet": # 根据配置选择模型
        model = SwinUNet()
    else:
        model = torch.hub.load( # 从 PyTorch Hub 加载标准 U-Net 模型
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=False,
        )
    # 从数据集中获取一个样本，得到输入尺寸
    sample_input, _ = next(iter(dataloaders["train"]))
    input_size = tuple(sample_input.shape[1:])  # (C, H, W)
    # 打印模型结构和参数量
    # input_size = (3, 224, 224)  # 假设输入尺寸为 (3, 224, 224)，可根据实际情况修改
    print_model_summary(model, input_size, device)

    train_and_evaluate(    # 训练和评估模型
        model,
        dataloaders,
        dataset_sizes,
        config.training.model,
        config.paths.saves_root,
        config.paths.weights_root,
        config.paths.plots_root,
        config.training.num_epochs,
        config.optimizer.lr,
        config.optimizer.weight_decay,
        config.lr_scheduler.step_size,
        config.lr_scheduler.gamma,
        device,
    )


if __name__ == "__main__":
    main()
