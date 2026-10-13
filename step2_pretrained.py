# -*- coding: utf-8 -*-
"""
第二步：在手动模型基础上加载 ImageNet 预训练参数

对应实验指导书第 2 节。要点：
  · 手动搭建的 AlexNetManual 层命名与 torchvision 官方完全一致，所以能 strict=True 加载官方权重
  · 原始 AlexNet 是 1000 类，我们只需把最后一层换成 2 类
  · 冻结卷积层，只训练最后的全连接层 —— 把 AlexNet 当成"特征提取器"
  · 为了和第一步做【控制变量对比】，epochs 和 lr 保持不变（30 / 1e-4），
    唯一变量就是"权重从随机初始化换成 ImageNet 预训练"

运行：python step2_pretrained.py
"""
import torch
import torch.nn as nn
from torchvision import models

from common import (MODELS_DIR, count_params, get_device, get_loaders,
                    print_banner, record_result, train_and_eval)


class AlexNetManual(nn.Module):
    """与 torchvision.models.alexnet 结构完全一致的手写版本。

    要点：卷积部分必须命名为 features、分类部分必须命名为 classifier，
    层顺序也要一致，否则加载官方权重时 key 对不上。
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),   # 索引 1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),          # 索引 4
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),   # 索引 6 ← 换这里
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_pretrained_alexnet(num_classes=2, freeze_conv=True):
    """加载 ImageNet 预训练 AlexNet，并把分类头换成 num_classes 类。"""
    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net = AlexNetManual(num_classes=1000)       # 1. 先建 1000 类的完整结构
    net.load_state_dict(official.state_dict())  # 2. 形状完全一致，strict=True 直接加载
    net.classifier[6] = nn.Linear(4096, num_classes)  # 3. 换成 2 类（新头随机初始化）

    if freeze_conv:
        # 冻结：只训练 classifier（也可以只训练最后一层）
        for name, param in net.named_parameters():
            param.requires_grad = name.startswith("classifier")
    return net


def main():
    print_banner("第二步：加载 ImageNet 预训练权重（冻结卷积层）")
    device = get_device()
    print(f"设备: {device}")

    train_loader, val_loader = get_loaders(batch_size=16)
    net = build_pretrained_alexnet(num_classes=2, freeze_conv=True)

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"模型参数量: {count_params(net):.1f} M"
          f"（其中可训练 {trainable / 1e6:.1f} M，只占 {trainable / total:.1%}）")
    print(f"训练集 {len(train_loader.dataset)} 张 / 验证集 {len(val_loader.dataset)} 张\n")

    EPOCHS, LR = 30, 1e-4          # 与第一步保持一致，保证只改了"是否预训练"这一个变量
    acc = train_and_eval(net, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), MODELS_DIR / "step2_alexnet_pretrained.pth")
    print(f"\n【第二步 预训练+冻结卷积】最终准确率: {acc:.4f}")

    record_result("step2_预训练AlexNet冻结卷积", {
        "accuracy": round(acc, 4), "epochs": EPOCHS, "lr": LR,
        "params_M": round(count_params(net), 1),
        "trainable_M": round(trainable / 1e6, 1),
        "pretrained": True, "frozen": "只训练 classifier",
        "weight": "models/step2_alexnet_pretrained.pth",
    })


if __name__ == "__main__":
    main()
