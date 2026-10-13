# -*- coding: utf-8 -*-
"""
第四步：调用 torchvision.models 封装好的模型横向对比

对应实验指导书第 4 节。对比三个模型（都加载 ImageNet 预训练、都只训练分类头）：
  resnet18 / mobilenet_v3_small / shufflenet_v2_x1_0

结论预告：小数据集上三者准确率通常相差不大（都在 0.9+），
此时【参数量 / 推理速度 / 模型体积】才是选型关键 —— 这正是第六步部署的前提。

运行：python step4_compare_models.py
"""
import time

import torch
import torch.nn as nn
from torchvision import models

from common import (MODELS_DIR, count_params, get_device, get_loaders,
                    print_banner, record_result, train_and_eval)

MODEL_NAMES = ["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0"]


def build_model(name, num_classes=2, freeze=True):
    """统一接口：加载预训练模型并替换分类头（不同模型的头名字不一样）"""
    constructors = {
        "resnet18":           lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT),
        "shufflenet_v2_x1_0": lambda: models.shufflenet_v2_x1_0(
            weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
    }
    net = constructors[name]()

    # 找到并替换分类头（不同模型的头名字不一样）
    if name == "resnet18":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    elif name == "mobilenet_v3_small":
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
        head_params = net.classifier[3].parameters()
    elif name == "shufflenet_v2_x1_0":
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        head_params = net.fc.parameters()
    else:
        raise ValueError(f"未知模型: {name}")

    if freeze:  # 冻结特征提取层，只训练分类头（CPU 友好）
        for p in net.parameters():
            p.requires_grad = False
        for p in head_params:
            p.requires_grad = True
    return net


@torch.no_grad()
def measure_inference_ms(net, device, size=224, rounds=20):
    """测单张图片的平均推理耗时（毫秒），用于选型参考。"""
    net.to(device).eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(3):          # 预热
        net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rounds):
        net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rounds * 1000


def main():
    print_banner("第四步：torchvision 官方模型横向对比")
    device = get_device()
    print(f"设备: {device}\n")

    train_loader, val_loader = get_loaders(batch_size=16)
    print(f"训练集 {len(train_loader.dataset)} 张 / 验证集 {len(val_loader.dataset)} 张\n")

    EPOCHS, LR = 5, 1e-3
    results = {}

    for name in MODEL_NAMES:
        print(f"\n===== 模型: {name} =====")
        net = build_model(name)
        t0 = time.perf_counter()
        acc = train_and_eval(net, train_loader, val_loader, epochs=EPOCHS, lr=LR,
                             device=device, verbose=False)
        train_sec = time.perf_counter() - t0
        infer_ms = measure_inference_ms(net, device)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), MODELS_DIR / f"model_{name}.pth")  # 第五步要用

        results[name] = {
            "accuracy": round(acc, 4),
            "params_M": round(count_params(net), 2),
            "train_sec": round(train_sec, 1),
            "sec_per_epoch": round(train_sec / EPOCHS, 2),
            "infer_ms": round(infer_ms, 1),
        }
        print(f"  准确率 {acc:.4f} | 参数量 {count_params(net):.2f} M | "
              f"训练 {train_sec:.1f}s（{train_sec / EPOCHS:.2f}s/轮） | "
              f"单张推理 {infer_ms:.1f} ms")

    print_banner("横向对比结果")
    print(f"{'模型':<22}{'参数量':>9}{'准确率':>9}{'每轮用时':>11}{'单张推理':>11}")
    print("-" * 66)
    for name, v in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"{name:<22}{v['params_M']:>7.2f}M{v['accuracy']:>9.4f}"
              f"{v['sec_per_epoch']:>10.2f}s{v['infer_ms']:>9.1f}ms")
    print("-" * 66)

    record_result("step4_官方模型横向对比", {
        "epochs": EPOCHS, "lr": LR, "device": str(device), "models": results,
    })


if __name__ == "__main__":
    main()
