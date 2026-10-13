# -*- coding: utf-8 -*-
"""
第三步：加入数据增强（第一 / 二 / 三步横向对比）

对应实验指导书第 3 节。要点：
  · 数据增强 = 不增加真实图片，人为造出"看起来不同"的训练样本，
    逼模型学本质特征而不是背图
  · 增强只作用于【训练集】，验证集必须是干净的 Resize + Normalize，
    否则评估指标不可信（指导书特意强调过）
  · 本步用 ResNet18Manual（手动搭建的 ResNet18）+ 预训练权重，只训练 fc 层

运行：python step3_augment.py
"""
import json

import torch
import torch.nn as nn
from torchvision import models, transforms

from common import (IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, OUTPUT_DIR,
                    RESULTS_FILE, count_params, get_device, get_loaders,
                    print_banner, record_result, train_and_eval)
from step1_scratch import ResNet18Manual

# 训练集增强：翻转 + 随机裁剪缩放 + 颜色抖动 + 小角度旋转
# 逐项对应现实中的拍摄变化（朝向/距离构图/光线白平衡/相机角度）
AUGMENT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),   # 随机裁剪缩放
    transforms.RandomHorizontalFlip(),                     # 随机水平翻转
    transforms.ColorJitter(0.2, 0.2, 0.2),                 # 随机亮度/对比度/饱和度
    transforms.RandomRotation(10),                         # 随机小角度旋转
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_resnet18_pretrained(num_classes=2, freeze_except_fc=True):
    """手动 ResNet18 结构 + 官方预训练权重；只训练 fc 层。

    ⚠️ 指导书里的写法在这里会报错：
        net = ResNet18Manual(num_classes=2)
        net.load_state_dict(official.state_dict(), strict=False)
    strict=False 只能忽略"多出来/缺失"的 key，**不能**解决形状不匹配，
    因为官方 fc 是 Linear(512,1000) 而我们的是 Linear(512,2)。

    正确做法（与第二步加载 AlexNet 时完全一致）：
        先建 1000 类的完整结构并把权重灌进去 → 再把分类头换成 2 类。
    """
    official = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net = ResNet18Manual(num_classes=1000)          # 1. 先建 1000 类的完整结构
    net.load_state_dict(official.state_dict())      # 2. 形状完全一致，strict=True 直接加载
    net.fc = nn.Linear(512, num_classes)            # 3. 换成 2 类（新头随机初始化）

    if freeze_except_fc:
        for name, param in net.named_parameters():
            param.requires_grad = "fc" in name
    return net


def print_comparison():
    """把前三步的结果拉出来做个横向对比表。"""
    if not RESULTS_FILE.exists():
        return
    data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    keys = ["step1_从零训练AlexNet", "step2_预训练AlexNet冻结卷积", "step3_预训练ResNet18加数据增强"]
    print("\n" + "-" * 66)
    print(f"{'实验':<34}{'准确率':>10}{'参数量':>12}")
    print("-" * 66)
    for k in keys:
        if k in data:
            v = data[k]
            print(f"{k:<34}{v['accuracy']:>10.4f}{v['params_M']:>10.1f} M")
    print("-" * 66)


def main():
    print_banner("第三步：预训练 ResNet18 + 数据增强")
    device = get_device()
    print(f"设备: {device}")

    train_loader, val_loader = get_loaders(batch_size=16, train_transform=AUGMENT_TRANSFORM)
    net = build_resnet18_pretrained(num_classes=2, freeze_except_fc=True)

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"模型参数量: {count_params(net):.1f} M"
          f"（其中可训练 {trainable / 1e6:.2f} M，只训练 fc 层）")
    print(f"训练集 {len(train_loader.dataset)} 张（带增强）/ 验证集 {len(val_loader.dataset)} 张（无增强）\n")

    EPOCHS, LR = 5, 1e-3
    acc = train_and_eval(net, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), MODELS_DIR / "step3_resnet18_augment.pth")
    print(f"\n【第三步 预训练+数据增强】最终准确率: {acc:.4f}")

    record_result("step3_预训练ResNet18加数据增强", {
        "accuracy": round(acc, 4), "epochs": EPOCHS, "lr": LR,
        "params_M": round(count_params(net), 1),
        "pretrained": True, "frozen": "只训练 fc",
        "augment": ["RandomResizedCrop", "RandomHorizontalFlip", "ColorJitter", "RandomRotation"],
        "weight": "models/step3_resnet18_augment.pth",
    })
    print_comparison()


if __name__ == "__main__":
    main()
