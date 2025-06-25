from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import albumentations as A


class MRI_Dataset(Dataset):

    def __init__(self, input_dir, label_dir, augmentation=None, transform=None):

        self.input_dir = input_dir # 输入图像路径列表
        self.label_dir = label_dir # 标签（掩码）路径列表
        self.transform = transform # 图像转换操作
        self.augmentation = augmentation# 数据增强操作

    def __len__(self):#直接返回输入图像路径列表的长度
        return len(self.input_dir)

    def __getitem__(self, idx):
        # 读取图像和掩码
        input_img = cv2.imread(self.input_dir[idx])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)# BGR 转 RGB

        label_img = cv2.imread(self.label_dir[idx], 0)# 以灰度模式读取掩码

        # input_m , input_s = np.mean(input_img, axis=(0, 1)), np.std(input_img, axis=(0, 1))
        # 定义图像转换操作
        input_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((256,256)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=input_m, std=input_s)
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        label_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.Resize((256,256)),
                transforms.ToTensor(),
            ]
        )
        # 定义数据增强操作
        augment_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.2),    # 水平翻转（20%概率）
                A.Rotate(limit=(-10, 10), p=0.2, border_mode=0),    # 旋转（±10度，20%概率）
                A.Affine(
                    scale=(0.9, 1.1), shear=(-15, 15), translate_percent=(0, 0.1), p=0.2    # 仿射变换
                ),
            ]
        )
        # 应用数据增强
        if self.augmentation:
            transformed = augment_transform(image=input_img, mask=label_img)
            input_img = transformed["image"]
            label_img = transformed["mask"]
        # 应用图像转换
        if self.transform:
            input_img = input_transform(input_img)
            label_img = label_transform(label_img)

        return input_img, label_img
