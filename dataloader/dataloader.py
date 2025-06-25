from torch.utils.data import DataLoader
from .dataset import MRI_Dataset
from .utils import (
    download_from_gdrive,
    get_patient_list,
    get_image_paths,
    split_dataset,
    unzip_dataset,
)
import os

#创建单个数据加载器，X, Y：图像路径列表和对应的掩码路径列表，augmentation：是否启用数据增强，transform：是否应用图像转换（如归一化）
def create_dataloader(
    X, Y, batch_size, shuffle=True, num_workers=2, augmentation=False, transform=True
):
    """
    Creates a dataloader for the given dataset.

    Args:
        X (list): List of image file paths.
        Y (list): List of corresponding mask file paths.
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle the dataset.是否打乱数据顺序
        num_workers (int): Number of workers for data loading.并行加载数据的进程数
        augmentation (bool): Whether to apply augmentation.
        transform (bool): Whether to apply transformation.

    Returns:
        DataLoader: PyTorch DataLoader object for the dataset.
    """
    dataset = MRI_Dataset(X, Y, augmentation=augmentation, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader, len(dataset)#PyTorch 数据加载器和数据集大小


def prepare_dataloaders(
    gdrive_url,
    dataset_zip_path,
    dataset_path,
    save_path,
    batch_size,
    augmentation,
    train_ratio,
    valid_ratio,
    seed,
    num_workers=2,
):
    """
    Generalized function to handle dataset downloading, processing, and dataloader preparation.

    Args:
        dataset_name (str): The Kaggle dataset name (e.g., "mateuszbuda/lgg-mri-segmentation").
        dataset_path (str): Path where the dataset is stored (or should be downloaded).
        save_path (str): Path to save preprocessed data.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of workers for data loading.
        train_ratio (float): Ratio for the training set.
        valid_ratio (float): Ratio for the validation set.
        seed (int): Random seed for reproducibility.
        augmentation (bool): Data augmentation.

    Returns:
        tuple: (dataloaders dictionary, dataset_sizes dictionary)
    """
    # Step 1: Download dataset if not present，下载和解压数据集
    if not os.path.exists(dataset_zip_path):
        download_from_gdrive(gdrive_url, dataset_zip_path)
        # download_dataset_kaggle(dataset_name, save_path=dataset_path)

    # if not os.listdir(dataset_path):
    #     unzip_dataset(dataset_zip_path, dataset_path)

    # Step 2: Retrieve patient list，获取患者列表并缓存
    patient_list = get_patient_list(dataset_path, save_path)

    # Step 3: Split dataset，划分患者数据为训练集、验证集
    train_list, valid_list, test_list = split_dataset(
        patient_list, train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed
    )
    # print(f"dataset_path:--{dataset_path}")
    # Step 4: Get image paths，获取各集合的图像和掩码路径
    X_train, Y_train = get_image_paths(dataset_path, train_list)
    X_valid, Y_valid = get_image_paths(dataset_path, valid_list)
    X_test, Y_test = get_image_paths(dataset_path, test_list)

    # Step 5: Create dataloaders，创建数据加载器
    dataloader_train, train_size = create_dataloader(
        X_train, Y_train, batch_size, num_workers=num_workers, augmentation=augmentation
    )
    dataloader_valid, valid_size = create_dataloader(
        X_valid, Y_valid, batch_size, num_workers=num_workers, augmentation=augmentation
    )
    dataloader_test, test_size = create_dataloader(
        X_test, Y_test, batch_size, num_workers=num_workers, augmentation=augmentation
    )

    dataloaders = {
        "train": dataloader_train,
        "val": dataloader_valid,
        "test": dataloader_test,
    }
    dataset_sizes = {"train": train_size, "val": valid_size, "test": test_size}

    print("✅ Dataloaders prepared successfully!")

    return dataloaders, dataset_sizes
