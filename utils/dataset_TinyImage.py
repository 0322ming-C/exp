from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import os.path as path
import torch
import csv
import os
import numpy as np
import PIL.Image as Image
import random

# 设置随机种子，保证可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class TINYIMAGENET(Dataset):
    def __init__(self,config:dict,
                size=(64, 64), set_name='train',
                isAugment=True):

        self.path_to_data = config["tinyimagenet_path"]
        self.mapping_name2id = {}
        self.mapping_id2name = {}
        with open(path.join(self.path_to_data, 'wnids.txt')) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            idx = 0
            for row in csv_reader:
                self.mapping_id2name[idx] = row[0]
                self.mapping_name2id[row[0]] = idx
                idx += 1

        # if set_name=='test':  set_name = 'val'

        self.size = size
        self.set_name = set_name
        self.path_to_data = config["tinyimagenet_path"]
        self.isAugment = isAugment

        self.imageNameList = []
        self.className = []
        self.labelList = []
        self.mappingLabel2Name = dict()
        curLabel = 0

        if self.set_name == 'test':
            img_dir = os.path.join(self.path_to_data, 'val', 'images')
            for file_name in os.listdir(img_dir):
                if file_name[-4:] == 'JPEG':
                    self.imageNameList += [path.join(self.path_to_data, 'val', 'images', file_name)]
                    self.labelList += [0]

        elif self.set_name == 'val':
            with open(path.join(self.path_to_data, 'val', 'val_annotations.txt')) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='\t')
                line_count = 0
                for row in csv_reader:
                    self.imageNameList += [path.join(self.path_to_data, 'val', 'images', row[0])]
                    self.labelList += [self.mapping_name2id[row[1]]]
                    # with open(path.join(self.path_to_data, 'val', 'val_annotations.txt')) as csv_file:
            #    csv_reader = csv.reader(csv_file, delimiter='\t')
            #    line_count = 0
            #    for row in csv_reader:
            #        self.imageNameList += [path.join(self.path_to_data, 'val', 'images', row[0])]
            #        self.labelList += [self.mapping_name2id[row[1]]]
        else:  # 'train'
            self.current_class_dir = path.join(self.path_to_data, self.set_name)
            for curClass in os.listdir(self.current_class_dir):
                if curClass[0] == '.':   continue

                curLabel = self.mapping_name2id[curClass]
                for curImg in os.listdir(path.join(self.current_class_dir, curClass, 'images')):
                    if curImg[0] == '.':    continue
                    self.labelList += [curLabel]
                    self.imageNameList += [path.join(self.path_to_data, self.set_name, curClass, 'images', curImg)]

        self.current_set_len = len(self.labelList)

        if self.set_name == 'test' or self.set_name == 'val' or not self.isAugment:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.size[0], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return self.current_set_len

    def __getitem__(self, idx):
        curLabel = np.asarray(self.labelList[idx])
        curLabel = torch.tensor(curLabel, dtype=torch.long)
        curImage = self.imageNameList[idx]
        curImage = Image.open(curImage).convert('RGB')
        curImage = self.transform(curImage)

        return curImage, curLabel

def load_tinyimagenet(config):
    print("加载TinyImageNet数据集...")

    # 创建数据集实例
    train_set = TINYIMAGENET(set_name='train', config=config, isAugment=True)
    val_set = TINYIMAGENET(set_name='val', config=config, isAugment=False)
    # test_set = TINYIMAGENET(set_name='test', config=config, isAugment=False)

    # 创建数据加载器
    # train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    return train_set, val_set#, test_loader


def load_open_dataset(dataset_name, config):
    """加载开集数据集"""
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "SVHN":
        dataset = datasets.SVHN(
            root="./data", split="train", download=True, transform=transform)
        test_dataset = datasets.SVHN(
            root="./data", split="test", download=True, transform=transform)
    elif dataset_name == "MNIST":
        # MNIST转换为三通道
        transform.transforms.insert(0, transforms.Grayscale(3))
        dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    return dataset, test_dataset


def load_data(config):
    """准备闭集和开集数据"""
    # 加载闭集数据 (TinyImageNet)
    closed_train, closed_test = load_tinyimagenet(config)

    # 划分闭集训练/验证 (80/20)
    closed_size = len(closed_train)
    val_size = int(0.2 * closed_size)
    train_size = closed_size - val_size

    indices = np.arange(closed_size)
    np.random.shuffle(indices)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    closed_trainset = Subset(closed_train, train_indices)
    closed_val = Subset(closed_train, val_indices)  # 注意：这是从训练集划分的验证集

    # 加载开集数据
    open_datasets = {}
    for name in config["open_datasets"]:
        train_set, test_set = load_open_dataset(name, config)
        open_datasets[name] = {
            "train": train_set,
            "test": test_set
        }

    return {
        "closed": {
            "train": closed_trainset,
            "val": closed_val,
            "test": closed_test
        },
        "open": open_datasets
    }