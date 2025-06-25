import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet_with_Residual, AttentionUNet, SwinUNet
from torchvision import transforms

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="脑肿瘤分割模型推理脚本")
    parser.add_argument("--input", "-i", type=str, default=r"input/TCGA_HT_A616_19991226",
                        help="输入图片路径或文件夹")
    parser.add_argument("--output", "-o", type=str, default="output/SwinUNet/TCGA_HT_A616_19991226",
                        help="输出结果保存路径")
    parser.add_argument("--weights", "-w", type=str, default="weights/SwinUNet/best_model.pt",
                        help="模型权重文件路径")
    parser.add_argument("--model", type=str, default="SwinUNet",
                        choices=[ "Unet_Basic","UNet_with_Residual", "AttentionUNet", "SwinUNet"],
                        help="模型类型")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二值化阈值")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备")
    return parser.parse_args()


def load_model(weights_path, model_type, device):
    """加载训练好的模型"""
    if model_type == "UNet_with_Residual":
        model = UNet_with_Residual()
    elif model_type == "AttentionUNet":  # 根据配置选择模型
        model = AttentionUNet()
    elif model_type == "SwinUNet": # 根据配置选择模型
        model = SwinUNet()
    else:
        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=False,
        )

    # 加载权重
    if device == "cuda":
        model   = torch.load(weights_path, weights_only=False)
    else:
        # 从GPU加载到CPU
        model   = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
    # 从检查点中提取模型权重
    # model.load_state_dict(checkpoint["model_state_dict"])
    # # 加载权重字典而非完整模型
    # state_dict = torch.load(weights_path, map_location=device)
    # model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    # # 打印加载信息
    # if "epoch" in checkpoint:
    #     print(f"加载 epoch {checkpoint['epoch']} 的模型权重")
    # if "loss" in checkpoint:
    #     print(f"验证损失: {checkpoint['loss']:.4f}")

    return model


def preprocess_image(image_path):
    """预处理输入图像"""
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换为PyTorch张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    img_tensor = transform(img).unsqueeze(0)  # 添加batch维度
    return img_tensor, img


def predict(model, input_tensor, device, threshold=0.5):
    """模型推理并生成掩码"""
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)

        # 应用阈值二值化
        mask = (output > threshold).float().cpu().squeeze(0).squeeze(0).numpy()

    return mask


def calculate_area(mask):
    """计算掩码的像素面积"""
    return np.count_nonzero(mask)


def visualize_results(input_img, mask, label_mask , output_path, area, label_area):
    """可视化结果并保存"""
    # 创建结果图像
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # 原图
    ax1.imshow(input_img)
    ax1.set_title("原始图像")
    ax1.axis("off")

    # # 掩码
    # ax2.imshow(mask, cmap="gray")
    # ax2.set_title("分割掩码")
    # ax2.axis("off")

    # 原始图像叠加标注掩码（蓝色）
    overlay_label = input_img.copy()
    if label_mask is not None:
        overlay_label[label_mask > 0.5] = [0, 255, 0]  # 在原图上叠加绿色标注掩码
    ax2.imshow(overlay_label)
    if label_area is not None:
        ax2.set_title(f"原始图像叠加标注掩码 (面积: {label_area} 像素)")
    else:
        ax2.set_title("原始图像叠加标注掩码 (无标注)")
    ax2.axis("off")

    # 叠加图
    overlay = input_img.copy()
    overlay[mask > 0.5] = [255, 0, 0]  # 在原图上叠加红色掩码
    ax3.imshow(overlay)
    ax3.set_title(f"掩码叠加 (面积: {area} 像素)")
    ax3.axis("off")

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 单独保存掩码图像
    mask_path = output_path.replace(".png", "_mask.png")
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    # if label_mask is not None:
    #     label_mask_path = output_path.replace(".png", "_label_mask.png")
    #     cv2.imwrite(label_mask_path, (label_mask * 255).astype(np.uint8))

def process_single_image(model, image_path, output_dir, device, threshold):
    """处理单张图像"""
    # 预处理
    input_tensor, input_img = preprocess_image(image_path)

    # 预测
    mask = predict(model, input_tensor, device, threshold)

    # 计算面积
    area = calculate_area(mask)

    # 读取标注好的掩码图像
    mask_path = image_path.replace(".tif", "_mask.tif")  # 假设掩码文件名是原文件名加 _mask.png
    if os.path.exists(mask_path):
        label_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        label_mask = (label_mask > 0).astype(np.uint8)
        label_area = calculate_area(label_mask)
    else:
        label_mask  = None
        label_area = None
        print(f"{mask_path}不存在")

    # 生成输出文件名
    file_name = os.path.basename(image_path)
    file_base, file_ext = os.path.splitext(file_name)
    output_path = os.path.join(output_dir, f"{file_base}_result.png")

    # 可视化并保存结果
    visualize_results(input_img, mask, label_mask , output_path, area , label_area )

    # 打印结果
    print(f"已处理: {image_path}")
    print(f"掩码面积: {area} 像素")
    if label_area is not None:
        print(f"标注掩码面积: {label_area} 像素")
    print(f"结果保存至: {output_path}")

    return output_path, area, label_area


def main():
    """主函数"""
    args = get_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载模型
    model = load_model(args.weights, args.model, args.device)
    print(f"已加载模型: {args.weights}")
    print(f"使用设备: {args.device}")

    # 处理输入
    if os.path.isfile(args.input):
        # 处理单张图像
        process_single_image(model, args.input, args.output, args.device, args.threshold)
    elif os.path.isdir(args.input):
        # 处理文件夹中的所有图像
        image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        image_paths = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.isfile(os.path.join(args.input, f)) and
               f.lower().endswith(tuple(image_extensions)) and
               not f.lower().endswith("_mask.tif")  # 排除掩码图像
        ]

        if not image_paths:
            print(f"错误: 目录 {args.input} 中未找到图像文件")
            return

        print(f"找到 {len(image_paths)} 张图像")

        # 处理每张图像
        results = []
        for i, img_path in enumerate(image_paths):
            print(f"\n处理图像 {i + 1}/{len(image_paths)}: {img_path}")
            output_path, area , label_area = process_single_image(
                model, img_path, args.output, args.device, args.threshold
            )
            results.append((img_path, output_path, area, label_area ))

        # 保存统计结果
        stats_path = os.path.join(args.output, "statistics.txt")
        with open(stats_path, "w") as f:
            f.write("图像分割统计结果\n")
            f.write("=" * 100 + "\n")
            f.write(f"总处理图像数: {len(results)}\n\n")
            f.write(f"{'原始图像':<40} {'推理掩码面积(像素)':<15} {'标注掩码面积(像素)':<15}{'结果路径'}\n")
            f.write("-" * 100 + "\n")
            for img_path, out_path, area , label_area  in results:
                label_area_str = str(label_area) if label_area is not None else "N/A"
                f.write(f"{os.path.basename(img_path):<40} {area:<15} {label_area_str:<15} {os.path.basename(out_path)}\n")

        print(f"\n所有图像处理完成，统计结果已保存至: {stats_path}")
    else:
        print(f"错误: 输入路径 {args.input} 不存在")


if __name__ == "__main__":
    main()