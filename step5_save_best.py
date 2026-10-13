# -*- coding: utf-8 -*-
"""
第五步（上）：训练过程中保存验证集最佳模型

对应实验指导书第 5.1 / 5.2 节。要点：
  · 只保存参数（state_dict），文件小、跨版本稳定；不推荐保存整个模型对象
  · 工程惯例：边训练边看验证集准确率，只保存历史最佳（早停思想的简化版）
  · 本步产出的 models/best_model.pth 供第六步部署使用

运行：python step5_save_best.py
"""
import torch
import torch.nn as nn
from torchvision import models

from common import (MODELS_DIR, count_params, get_device, get_loaders,
                    print_banner, record_result)

BEST_PATH_NAME = "best_model.pth"


def train_with_best_save(net, train_loader, val_loader, epochs=5, lr=1e-3,
                         save_path="best_model.pth", device=None):
    device = device or get_device()
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)

    best_acc = 0.0
    history = []
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(X), Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        # 验证
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for X, Y in val_loader:
                correct += (net(X.to(device)).argmax(1) == Y.to(device)).sum().item()
                total += Y.size(0)
        acc = correct / total
        history.append(round(acc, 4))
        print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: "
              f"{running_loss / len(train_loader.dataset):.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:                      # 只保留历史最佳
            best_acc = acc
            torch.save(net.state_dict(), save_path)
            print(f"  ^ 最佳模型已保存到 {save_path}")

    print(f"最佳准确率: {best_acc:.4f}")
    return best_acc, history


def main():
    print_banner("第五步：保存验证集最佳模型")
    device = get_device()
    print(f"设备: {device}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / BEST_PATH_NAME

    train_loader, val_loader = get_loaders(batch_size=16)
    print(f"训练集 {len(train_loader.dataset)} 张 / 验证集 {len(val_loader.dataset)} 张\n")

    # 用第四步里表现最稳的 resnet18 + 预训练权重，只训练分类头
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 2)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True

    print(f"模型参数量: {count_params(net):.1f} M（只训练 fc 层）\n")

    EPOCHS, LR = 5, 1e-3
    best_acc, history = train_with_best_save(net, train_loader, val_loader,
                                             epochs=EPOCHS, lr=LR,
                                             save_path=str(save_path), device=device)

    record_result("step5_保存最佳模型", {
        "best_accuracy": round(best_acc, 4), "epochs": EPOCHS, "lr": LR,
        "val_acc_history": history,
        "weight": "models/best_model.pth",
        "note": "只保存参数（state_dict），加载时先建同结构模型再 load_state_dict",
    })
    print(f"\n权重文件: {save_path}")


if __name__ == "__main__":
    main()
