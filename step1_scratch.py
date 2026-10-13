# -*- coding: utf-8 -*-
"""
第一步：手动搭建模型，从零训练（以 AlexNet 为例）

对应实验指导书第 1 节。要点：
  · 网络结构手写，不使用任何预训练权重（随机初始化）
  · 用 xavier_uniform_ 初始化权重
  · 预期结果 0.65 ~ 0.80：小数据集上从零训练学不出通用视觉特征

本模块同时提供【手动搭建的两种网络】，供后续步骤复用：
  · AlexNet        —— 第一步从零训练用
  · ResNet18Manual —— 第三步用（指导书里就是 `from step1_scratch import ResNet18Manual`）
两者的层命名都与 torchvision 官方实现完全一致，因此可以 strict=True 加载官方预训练权重。

运行：python step1_scratch.py
"""
import torch
import torch.nn as nn

from common import (MODELS_DIR, count_params, get_device, get_loaders,
                    print_banner, record_result, train_and_eval)


class AlexNet(nn.Module):
    """精简版 AlexNet（去掉了当年双 GPU 并行的设计）"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            # 11*11 大窗口捕捉对象，步幅 4 快速降维，输出通道数远大于 LeNet
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 窗口变小，填充保持高宽一致，通道数继续增大
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 三个连续的小窗口卷积层，通道数进一步增加
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            # 全连接层用 dropout 减轻过拟合（样本太少，这层很关键）
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    """ResNet 的基本残差块：两层 3×3 卷积 + 一条"抄近道"的恒等映射"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)     # 形状不一致时用 1×1 卷积对齐通道
        out = out + identity                  # ← 残差连接：梯度可以"抄近道"
        return self.relu(out)


class ResNet18Manual(nn.Module):
    """手动搭建的 ResNet18（层命名与 torchvision 一致，可加载官方预训练权重）"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)


def main():
    print_banner("第一步：手动搭建 AlexNet，从零训练（无预训练权重）")
    device = get_device()
    print(f"设备: {device}")

    train_loader, val_loader = get_loaders(batch_size=16)
    net = AlexNet(num_classes=2)
    net.apply(init_weight)

    print(f"模型参数量: {count_params(net):.1f} M")
    print(f"训练集 {len(train_loader.dataset)} 张 / 验证集 {len(val_loader.dataset)} 张\n")

    EPOCHS, LR = 30, 1e-4
    acc = train_and_eval(net, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), MODELS_DIR / "step1_alexnet_scratch.pth")
    print(f"\n【第一步 从零训练】最终准确率: {acc:.4f}")

    record_result("step1_从零训练AlexNet", {
        "accuracy": round(acc, 4), "epochs": EPOCHS, "lr": LR,
        "params_M": round(count_params(net), 1), "pretrained": False,
        "weight": "models/step1_alexnet_scratch.pth",
    })


if __name__ == "__main__":
    main()
