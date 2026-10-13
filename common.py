# -*- coding: utf-8 -*-
"""
common.py —— 公共代码（对应实验指导书第 0 节）

所有实验都从这里导入，保证：
  ① 同一份数据划分（随机种子 42，80% 训练 / 20% 验证）
  ② 同一套评价指标（准确率 / 精确率 / 召回率 / F1）
这样每次只改一个变量（网络结构 / 预训练 / 数据增强），对比才公平。
"""
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 项目根目录（本文件所在位置），所有相对路径都以它为基准
ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

# 标签顺序：cat -> 0，dog -> 1（全项目统一，不要改动）
CLASSES = ["cat 猫", "dog 狗"]


def get_device() -> torch.device:
    """有显卡用显卡，没有就退回 CPU。指导书里写的是 cpu，这里自动选择。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class FileNameDataset(Dataset):
    """从文件名解析标签：cat.x.jpg -> 0，dog.x.jpg -> 1"""

    def __init__(self, image_dir, transform=None):
        self.image_dir = str(image_dir)
        self.transform = transform
        self.images = sorted(
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not self.images:
            raise FileNotFoundError(f"目录里没有图片: {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        # 只用文件名判断标签（不能用整条路径：父目录名里也可能含 cat/dog）
        label = 0 if "cat" in img_name.lower() else 1
        if self.transform:
            image = self.transform(image)
        return image, label


def default_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def default_val_transform():
    # 验证集必须用干净的预处理，不做任何增强，否则指标不可信
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_loaders(data_dir=None, batch_size=16, train_transform=None, seed=42):
    """统一划分：80% 训练 / 20% 验证，固定随机种子保证各实验可比"""
    data_dir = Path(data_dir) if data_dir else DATASETS_DIR
    if train_transform is None:
        train_transform = default_train_transform()
    val_transform = default_val_transform()

    n_total = len(FileNameDataset(data_dir))
    train_size = int(0.8 * n_total)
    val_size = n_total - train_size

    # 训练集与验证集用同一个划分索引，但套不同的 transform
    train_set = FileNameDataset(data_dir, transform=train_transform)
    val_set = FileNameDataset(data_dir, transform=val_transform)

    g = torch.Generator().manual_seed(seed)
    idx_train, idx_val = random_split(range(n_total), [train_size, val_size], generator=g)

    train_loader = DataLoader(
        torch.utils.data.Subset(train_set, list(idx_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        torch.utils.data.Subset(val_set, list(idx_val)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# 训练 + 评估
# --------------------------------------------------------------------------- #
def train_and_eval(net, train_loader, val_loader, epochs=5, lr=1e-3, device=None, verbose=True):
    """训练并在验证集上评估，返回准确率（同时打印精确率/召回率/F1）。"""
    device = device or get_device()
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)

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
        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: "
                  f"{running_loss / len(train_loader.dataset):.4f}")

    return evaluate(net, val_loader, device)


@torch.no_grad()
def evaluate(net, val_loader, device=None, verbose=True):
    """在验证集上算指标。"""
    device = device or get_device()
    net.eval()
    all_preds, all_labels = [], []
    for X, Y in val_loader:
        preds = net(X.to(device)).argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(Y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)
    if verbose:
        print(f"准确率: {acc:.4f} | 精确率: {p:.4f} | 召回率: {r:.4f} | F1: {f1:.4f}")
    return acc


def count_params(net) -> float:
    """参数量（百万），用于第四步横向对比。"""
    return sum(p.numel() for p in net.parameters()) / 1e6


# --------------------------------------------------------------------------- #
# 结果汇总（六个步骤的结果都写进 output/results.json，方便最后做对比表）
# --------------------------------------------------------------------------- #
RESULTS_FILE = OUTPUT_DIR / "results.json"


def record_result(step: str, payload: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if RESULTS_FILE.exists():
        try:
            data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[step] = payload
    RESULTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[记录] {step} -> {RESULTS_FILE}")


def print_banner(title: str):
    print("\n" + "=" * 66)
    print(f"  {title}")
    print("=" * 66)

