import torch
from torchvision.models import resnet18
import torch.nn as nn
import torchvision
import os
import torch.optim as optim
from tqdm import tqdm
from .eval_funcs import evaluate_classifier
from .visualization import visualize_classifier_acc


class MNISTClassifier(nn.Module):
    """MNIST闭集分类器（基于ResNet18）"""

    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)

        # 适配MNIST输入尺寸
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        # 替换最后一层
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """提取特征（移除分类层）"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


def train_classifierI(train_loader, val_loader, config):
    print("\n训练闭集分类器...")
    classifier = MNISTClassifier(num_classes=len(config["closed_classes"])).to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config["lr_classifier"])

    best_acc = 0
    train_losses, val_accs = [], []

    for epoch in range(config["classifier_epochs"]):
        classifier.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"分类器 Epoch {epoch + 1}"):
            images = images.to(config["device"])
            labels = labels.to(config["device"])

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证集评估
        val_acc = evaluate_classifier(classifier, val_loader["closed"], config)
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}: 训练损失={train_loss:.4f}, 验证准确率={val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(),
                       os.path.join(config["save_dir"], "best_classifierI.pth"))
    visualize_classifier_acc(train_losses, val_accs, config)
    return classifier

class TinyImageClassifier(nn.Module):

    def __init__(self, num_classes=200):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)


    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        modules = list(self.backbone.children())[:-1]  # 去掉fc
        features = nn.Sequential(*modules)(x)  # [B, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 512]
        return features

def train_classifierII(train_loader, val_loader, config):
    print("\n训练闭集分类器...")
    classifier = TinyImageClassifier(num_classes=config["num_closed_classes"]).to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config["classifier_lr"], weight_decay=config["classifier_weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_acc = 0
    train_losses, val_accs = [], []

    for epoch in range(config["classifier_epochs"]):
        classifier.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"分类器 Epoch {epoch + 1}/{config['classifier_epochs']}"):
            images, labels = images.to(config["device"]), labels.to(config["device"])

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证集评估
        val_acc = evaluate_classifier(classifier, val_loader, config)
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        scheduler.step()

        print(f"Epoch {epoch + 1}: 训练损失={train_loss:.4f}, 验证准确率={val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(),
                       os.path.join(config["save_dir"], "best_classifierII.pth"))
    visualize_classifier_acc(train_losses, val_accs, config)
    return classifier







