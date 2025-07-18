import torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import numpy as np


def load_MNIST(config):
    print("加载MNIST数据集...")
    # MNIST数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 单通道转三通道
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 加载完整训练集
    full_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)

    # 划分闭集/开集索引
    closed_idx = [i for i, (_, label) in enumerate(full_set)
                  if label in config["closed_classes"]]
    open_idx = [i for i, (_, label) in enumerate(full_set)
                if label in config["open_classes"]]

    # 划分训练/验证集
    np.random.shuffle(closed_idx)
    val_size = int(len(closed_idx) * config["val_ratio"])
    train_idx = closed_idx[val_size:]
    val_closed_idx = closed_idx[:val_size]
    val_open_idx = open_idx[:val_size]  # 等量开集验证样本

    # 创建数据加载器
    train_loader = DataLoader(
        Subset(full_set, train_idx),
        batch_size=config["batch_size"], shuffle=True)

    val_loader = {
        "closed": DataLoader(
            Subset(full_set, val_closed_idx),
            batch_size=config["batch_size"]),
        "open": DataLoader(
            Subset(full_set, val_open_idx),
            batch_size=config["batch_size"])
    }

    # 加载测试集，全量数据做验证
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_closed_idx = [i for i, (_, label) in enumerate(test_set)
                       if label in config["closed_classes"]]
    test_open_idx = [i for i, (_, label) in enumerate(test_set)
                     if label in config["open_classes"]]

    test_loader = {
        "closed": DataLoader(
            Subset(test_set, test_closed_idx),
            batch_size=config["batch_size"]),
        "open": DataLoader(
            Subset(test_set, test_open_idx),
            batch_size=config["batch_size"])
    }

    print(f"训练集: {len(train_idx)}闭集样本")
    print(f"验证集: {len(val_closed_idx)}闭集 + {len(val_open_idx)}开集样本")
    print(f"测试集: {len(test_closed_idx)}闭集 + {len(test_open_idx)}开集样本")

    return train_loader, val_loader, test_loader













