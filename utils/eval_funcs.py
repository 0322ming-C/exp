from sklearn.metrics import roc_auc_score
import numpy as np
import torch

def evaluate_opengan(discriminator, classifier, data_loader_closed, data_loader_open, config):
    """评估开集检测性能（AUROC）"""
    discriminator.eval()
    all_scores = []
    all_labels = []

    # 处理闭集数据
    for img, _ in data_loader_closed:
        img = img.to(config["device"])
        with torch.no_grad():
            feats = classifier.extract_features(img)
            # scores = discriminator(feats).squeeze().detach().cpu().numpy()
            scores = discriminator(feats).detach().cpu().numpy()
            scores = np.asarray(scores).reshape(-1)
            if scores.shape[0] != len(img):
                scores = np.repeat(scores, len(img))

            # 检查NaN值并替换
            if np.isnan(scores).any():
                scores = np.nan_to_num(scores, nan=0.0)
                print("警告：检测到NaN值(闭集)，已替换为0.0")
        if isinstance(scores, np.ndarray) and scores.ndim == 0:
            scores = np.expand_dims(scores, 0)
        all_scores.extend(scores)
        all_labels.extend([0] * len(img))  # 闭集标签=0

    # 处理开集数据
    for img, _ in data_loader_open:
        img = img.to(config["device"])
        with torch.no_grad():
            feats = classifier.extract_features(img)
            scores = discriminator(feats).detach().cpu().numpy()
            scores = np.asarray(scores).reshape(-1)
            if scores.shape[0] != len(img):
                scores = np.repeat(scores, len(img))

            # 检查NaN值并替换
            if np.isnan(scores).any():
                scores = np.nan_to_num(scores, nan=0.0)
                print("警告：检测到NaN值(开集)，已替换为0.0")
        if isinstance(scores, np.ndarray) and scores.ndim == 0:
            scores = np.expand_dims(scores, 0)
        all_scores.extend(scores)
        all_labels.extend([1] * len(img))  # 开集标签=1

    # 过滤无效值
    all_scores = np.asarray(all_scores).reshape(-1)
    all_labels = np.asarray(all_labels).reshape(-1)
    valid_indices = ~np.isnan(all_scores)
    filtered_scores = all_scores[valid_indices]
    filtered_labels = all_labels[valid_indices]

    # 计算AUROC
    if len(np.unique(filtered_labels)) < 2:
        print("警告：验证集只有一个类别，无法计算AUROC")
        return 0.5

    try:
        return roc_auc_score(filtered_labels, filtered_scores)
    except ValueError:
        print("警告：AUROC计算错误，返回默认值0.5")
        return 0.5

def evaluate_classifier(classifier, data_loader, config):
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(config["device"])
            labels = labels.to(config["device"])

            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total