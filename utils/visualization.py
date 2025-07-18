import torch
from matplotlib import pyplot as plt
import os
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

from .eval_funcs import *


def visualize_classifier_acc(train_losses, val_accs, config):
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.title('分类器训练损失')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='验证准确率')
    plt.title('分类器验证准确率')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config["save_dir"], "classifier_training.png"))
    plt.close()


def visualize_opengan_training(train_d_losses, train_g_losses, val_aurocs, config):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_d_losses, label='判别器损失')
    plt.plot(train_g_losses, label='生成器损失')
    plt.title('OpenGAN训练损失')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_aurocs, label='验证集AUROC')
    plt.title('验证集性能')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config["save_dir"], "opengan_training.png"))
    plt.close()


def visualize_results(classifier, discriminator, test_loader_closed, test_loader_open, config):
    print("\n可视化结果...")
    # 生成示例图像
    # generator = Generator().to(config["device"])
    # z = torch.randn(16, 64).to(config["device"])
    # with torch.no_grad():
    #     fake_features = generator(z)

    # 在测试集上评估
    test_auroc = evaluate_opengan(discriminator, classifier, test_loader_closed, test_loader_open, config)
    print(f"测试集AUROC: {test_auroc:.4f}")

    # 可视化生成的特征空间
    visualize_feature_space(classifier, test_loader_closed, test_loader_open, config)

    # 可视化判别器决策边界
    visualize_decision_boundary(discriminator, classifier, test_loader_closed, test_loader_open, config)


def visualize_feature_space(classifier, data_loader_closed, data_loader_open, config):
    from sklearn.manifold import TSNE

    print("可视化特征空间...")
    # 收集特征和标签
    all_features = []
    all_labels = []

    # 闭集数据
    for img, labels in data_loader_closed:
        img = img.to(config["device"])
        with torch.no_grad():
            feats = classifier.extract_features(img)
        all_features.append(feats.cpu().numpy())
        all_labels.extend(labels.numpy())

    # 开集数据
    for img, labels in data_loader_open:
        img = img.to(config["device"])
        with torch.no_grad():
            feats = classifier.extract_features(img)
        all_features.append(feats.cpu().numpy())
        all_labels.extend([10] * len(img))  # 开集标签=10

    all_features = np.concatenate(all_features, axis=0)

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)

    # 绘制结果
    plt.figure(figsize=(10, 8))
    for label in range(11):  # 0-9为闭集数字，10为开集
        if label == 10:
            mask = [l == 10 for l in all_labels]
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        label='开集', alpha=0.6, marker='x')
        elif label in config["closed_classes"]:
            mask = [l == label for l in all_labels]
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        label=f'闭集{label}', alpha=0.6)

    plt.title('特征空间可视化 (t-SNE)')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(config["save_dir"], "feature_space.png"))
    plt.close()


def visualize_decision_boundary(discriminator, classifier, data_loader_closed, data_loader_open, config):
    print("可视化决策边界...")
    # 收集特征和标签
    features = []
    labels = []

    # 只取少量样本
    for img, lbl in data_loader_closed:
        img = img.to(config["device"])
        with torch.no_grad():
            feats = classifier.extract_features(img)
            scores = discriminator(feats).squeeze().cpu().numpy()

            # 检查NaN值
            if np.isnan(scores).any():
                continue

        features.append(feats.cpu().numpy())
        labels.extend([0] * len(img))  # 闭集标签=0
        if len(features) > 10:  # 限制样本数量
            break

    for img, lbl in data_loader_open:
        img = img.to(config["device"])
        with torch.no_grad():
            feats = classifier.extract_features(img)
            scores = discriminator(feats).squeeze().cpu().numpy()

            # 检查NaN值
            if np.isnan(scores).any():
                continue

        features.append(feats.cpu().numpy())
        labels.extend([1] * len(img))  # 开集标签=1
        if len(features) > 20:  # 限制样本数量
            break

    if len(features) == 0:
        print("警告：没有有效特征用于可视化决策边界")
        return

    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # 创建网格
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 预测网格点
    grid_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    grid_points = torch.tensor(grid_points, dtype=torch.float32).to(config["device"])

    with torch.no_grad():
        Z = discriminator(grid_points).cpu().numpy()

    # 检查NaN值
    if np.isnan(Z).any():
        print("警告：决策边界预测包含NaN值，无法可视化")
        return

    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap="RdBu")
    plt.colorbar()

    # 绘制样本点
    plt.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1],
                c='blue', edgecolors='k', label='闭集')
    plt.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1],
                c='red', edgecolors='k', marker='x', label='开集')

    plt.title('判别器决策边界')
    plt.xlabel('PCA维度1')
    plt.ylabel('PCA维度2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config["save_dir"], "decision_boundary.png"))
    plt.close()
