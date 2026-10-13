# -*- coding: utf-8 -*-
"""
第五步（下）：单张图片推理测试

对应实验指导书第 5.3 节。要点：
  · 训练与推理分离：推理时只需要【模型结构 + 权重】，不碰训练代码
  · 载入权重时：先建同结构模型 → 再 load_state_dict（对应第五步只存参数的做法）
  · 预处理必须与训练时【完全一致】：Resize((224,224)) + ToTensor + ImageNet 归一化

运行：
  python step5_test.py                                  # 用默认权重和默认图片
  python step5_test.py models/best_model.pth datasets/dog.5.jpg
  python step5_test.py models/best_model.pth datasets   # 传文件夹则批量预测
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from common import CLASSES, DATASETS_DIR, MODELS_DIR, get_device

PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(weight_path="best_model.pth", device=None):
    """先建同结构模型，再加载参数（对应第五步"只保存参数"的做法）。"""
    device = device or get_device()
    net = models.resnet18(weights=None)     # 结构一致即可，不用再下载预训练权重
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    net.to(device).eval()
    return net


@torch.no_grad()
def predict(image_path, net, device=None):
    device = device or next(net.parameters()).device
    img = Image.open(image_path).convert("RGB")
    x = PREDICT_TRANSFORM(img).unsqueeze(0).to(device)   # 增加 batch 维 -> [1,3,224,224]
    probs = torch.softmax(net(x), dim=1)[0]              # 得分转概率
    pred = int(probs.argmax())
    return CLASSES[pred], float(probs[pred]), probs.cpu()


def main():
    weight = sys.argv[1] if len(sys.argv) > 1 else str(MODELS_DIR / "best_model.pth")
    target = sys.argv[2] if len(sys.argv) > 2 else str(DATASETS_DIR / "cat.1.jpg")

    if not Path(weight).exists():
        print(f"[错误] 找不到权重文件: {weight}\n请先运行 python step5_save_best.py")
        return 1

    device = get_device()
    net = load_model(weight, device)
    print(f"权重: {weight}    设备: {device}\n")

    target_path = Path(target)

    # 情况一：单张图片
    if target_path.is_file():
        label, conf, probs = predict(target_path, net, device)
        print(f"图片: {target_path}")
        print(f"预测: {label}，置信度 {conf:.2%}")
        print(f"各类概率: 猫 {probs[0]:.2%} | 狗 {probs[1]:.2%}")
        return 0

    # 情况二：整个文件夹，批量跑一遍
    if target_path.is_dir():
        files = sorted(p for p in target_path.rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not files:
            print(f"[错误] 文件夹里没有图片: {target_path}")
            return 1
        print(f"{'文件名':<26}{'预测':<10}{'置信度':>10}")
        print("-" * 48)
        correct = 0
        for p in files:
            label, conf, _ = predict(p, net, device)
            truth = "cat" if "cat" in p.name.lower() else "dog"
            correct += truth in label
            flag = "" if conf >= 0.8 else "  <- 低置信度"
            print(f"{p.name[:25]:<26}{label:<10}{conf:>9.2%}{flag}")
        print("-" * 48)
        print(f"共 {len(files)} 张，准确率 {correct}/{len(files)} = {correct / len(files):.2%}")
        return 0

    print(f"[错误] 路径不存在: {target_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
