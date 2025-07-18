import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from .eval_funcs import evaluate_opengan
import os
from .visualization import visualize_opengan_training

class Discriminator(nn.Module):

    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            # fc(D → 64 * 8)
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            # fc(64 * 8 → 64 * 4)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            # fc(64 * 4 → 64 * 2)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            # fc(64 * 2 → 64 * 1)
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            # fc(64 * 1 → 1)
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):

    def __init__(self, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            # fc(64 → 64 * 8)
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            # fc(64 * 8 → 64 * 4)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            # fc(64 * 4 → 64 * 2)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            # fc(64 * 2 → 64 * 4)
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            # fc(64 * 4 → D)
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

def train_opengan(classifier, train_loader, val_loader, config):
    print("\n训练OpenGAN...")
    # 初始化模型
    generator = Generator(output_dim=config["feature_dim"]).to(config["device"])
    discriminator = Discriminator(input_dim=config["feature_dim"]).to(config["device"])

    # 优化器
    opt_g = optim.Adam(generator.parameters(), lr=config["lr_gan"], betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=config["lr_gan"], betas=(0.5, 0.999))

    # 添加梯度裁剪
    max_grad_norm = 1.0

    # 训练循环
    best_auroc = 0
    train_d_losses, train_g_losses, val_aurocs = [], [], []

    for epoch in range(config["gan_epochs"]):
        generator.train()
        discriminator.train()
        d_losses, g_losses = [], []

        for real_imgs, _ in tqdm(train_loader, desc=f"OpenGAN Epoch {epoch + 1}"):
            real_imgs = real_imgs.to(config["device"])

            # 提取真实闭集特征
            with torch.no_grad():
                real_features = classifier.extract_features(real_imgs)  #引入OTS特征

            # ===== 训练判别器 =====
            opt_d.zero_grad()

            # 真实样本损失 - 添加数值稳定性处理
            real_preds = discriminator(real_features)
            loss_real = -torch.log(torch.clamp(real_preds, min=1e-7)).mean()
            # loss_real = torch.log(torch.clamp(real_preds, min=1e-7)).mean()

            # 生成假样本
            batch_size = real_imgs.size(0)
            z = torch.randn(batch_size, 64).to(config["device"])
            fake_features = generator(z)
            fake_preds = discriminator(fake_features.detach())

            # 假样本损失 - 添加数值稳定性处理
            loss_fake = torch.log(torch.clamp(1 - fake_preds, min=1e-7)).mean()

            # 总判别器损失
            loss_d = loss_real + config["lambda_g"] * loss_fake
            # loss_d = -loss_real - config["lambda_g"] * loss_fake
            loss_d.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)

            opt_d.step()
            d_losses.append(loss_d.item())

            # ===== 训练生成器 =====
            opt_g.zero_grad()
            fake_preds = discriminator(fake_features)

            # 添加数值稳定性处理
            loss_g = -torch.log(torch.clamp(1 - fake_preds, min=1e-7)).mean()
            loss_g.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)

            opt_g.step()
            g_losses.append(loss_g.item())

        # 计算平均损失
        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)
        train_d_losses.append(avg_d_loss)
        train_g_losses.append(avg_g_loss)

        # 验证集评估
        val_auroc = evaluate_opengan(discriminator, classifier, val_loader["closed"], val_loader["open"], config)
        val_aurocs.append(val_auroc)

        print(f"Epoch {epoch + 1}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}, Val AUROC={val_auroc:.4f}")

        # 保存最佳判别器
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(discriminator.state_dict(),
                       os.path.join(config["save_dir"], "best_discriminator.pth"))
    visualize_opengan_training(train_d_losses, train_g_losses, val_aurocs, config)
    print(f"最佳验证AUROC: {best_auroc:.4f}")
    return discriminator


def train_openganII(classifier, train_loader, val_loader, config):
    """训练OpenGAN"""
    print("训练OpenGAN...")
    # 初始化模型
    generator = Generator(output_dim=config["feature_dim"]).to(config["device"])
    discriminator = Discriminator(input_dim=config["feature_dim"]).to(config["device"])

    # 优化器
    opt_g = optim.Adam(generator.parameters(), lr=config["opengan_lr"], betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=config["opengan_lr"], betas=(0.5, 0.999))

    # 训练循环
    best_auroc = 0.5
    d_losses, g_losses, aurocs = [], [], []

    for epoch in range(config["opengan_epochs"]):
        generator.train()
        discriminator.train()
        d_losses, g_losses = [], []

        for real_imgs, _ in tqdm(train_loader, desc=f"OpenGAN Epoch {epoch + 1}"):
            real_imgs = real_imgs.to(config["device"])

            # 提取真实闭集特征
            with torch.no_grad():
                real_features = classifier.extract_features(real_imgs)

            # ===== 训练判别器 =====
            opt_d.zero_grad()

            # 真实样本损失 - 添加数值稳定性处理
            real_preds = discriminator(real_features)
            loss_real = -torch.log(torch.clamp(real_preds, min=1e-7)).mean()

            # 生成假样本
            batch_size = real_imgs.size(0)
            z = torch.randn(batch_size, 64).to(config["device"])
            fake_features = generator(z)
            fake_preds = discriminator(fake_features.detach())

            # 假样本损失 - 添加数值稳定性处理
            loss_fake = torch.log(torch.clamp(1 - fake_preds, min=1e-7)).mean()

            # 总判别器损失
            loss_d = loss_real + config["lambda_g"] * loss_fake
            loss_d.backward()
            opt_d.step()
            d_losses.append(loss_d.item())

            # ===== 训练生成器 =====
            opt_g.zero_grad()
            fake_preds = discriminator(fake_features)

            # 添加数值稳定性处理
            loss_g = -torch.log(torch.clamp(1 - fake_preds, min=1e-7)).mean()
            loss_g.backward()
            opt_g.step()
            g_losses.append(loss_g.item())

        # 计算平均损失
        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        # 验证集评估
        val_auroc = evaluate_opengan(discriminator, classifier, val_loader["closed"], val_loader["open"], config)
        aurocs.append(val_auroc)

        print(f"Epoch {epoch + 1}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}, Val AUROC={val_auroc:.4f}")

        # 保存最佳判别器
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(discriminator.state_dict(),
                       os.path.join(config["output_dir"], "best_discriminator.pth"))
    visualize_opengan_training(d_losses, g_losses, aurocs, config)
    return discriminator