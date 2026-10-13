# -*- coding: utf-8 -*-
"""
make_figures.py —— 生成实验报告用的两张图

  1. output/experiment_summary.png  六个步骤的准确率对比 + 各模型横向对比
  2. output/independent_test.png    独立测试集 20 张的预测可视化（红色 = 误判）

独立测试集的 20 张图片来自公开的 Kaggle PetImages 数据集（Cat/Dog 索引 9000~9009），
与 datasets/ 里的训练数据完全不重叠，因此能真实反映泛化能力。

运行：python make_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from common import MODELS_DIR, OUTPUT_DIR, ROOT, get_device, print_banner
from step5_test import load_model, predict

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TEST_DIR = ROOT / "test_images"


def fig_experiment_summary():
    data = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))

    steps = [
        ("step1_从零训练AlexNet", "① 从零训练\nAlexNet"),
        ("step2_预训练AlexNet冻结卷积", "② +ImageNet预训练\n(冻结卷积)"),
        ("step3_预训练ResNet18加数据增强", "③ +数据增强\n(ResNet18)"),
        ("step4_官方模型横向对比", "④ 官方模型\n对比(resnet18)"),
    ]
    names, accs = [], []
    for key, label in steps:
        if key not in data:
            continue
        names.append(label)
        if key == "step4_官方模型横向对比":
            accs.append(data[key]["models"]["resnet18"]["accuracy"])
        else:
            accs.append(data[key]["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- 左图：六步实验准确率 ----
    colors = ["#c9c4bb", "#8fb3d9", "#5b8fc9", "#2f6ba8"]
    bars = axes[0].bar(names, accs, color=colors[:len(names)], width=0.62)
    for b, a in zip(bars, accs):
        axes[0].text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.1%}",
                     ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("验证集准确率")
    axes[0].set_title("实验步骤准确率对比", fontsize=13)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].axhline(1.0, color="#999", linestyle=":", linewidth=1)

    # ---- 右图：第四步多模型横向对比 ----
    if "step4_官方模型横向对比" in data:
        models_data = data["step4_官方模型横向对比"]["models"]
        m_names = list(models_data.keys())
        m_acc = [models_data[m]["accuracy"] for m in m_names]
        m_param = [models_data[m]["params_M"] for m in m_names]
        m_ms = [models_data[m]["infer_ms"] for m in m_names]

        y = range(len(m_names))
        bars = axes[1].barh(list(y), m_acc, color="#5b8fc9", height=0.55)
        axes[1].set_yticks(list(y))
        axes[1].set_yticklabels(
            [f"{n}\n{m_param[i]:.1f}M 参数 / {m_ms[i]:.1f}ms" for i, n in enumerate(m_names)],
            fontsize=9.5)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1.15)
        axes[1].set_xlabel("验证集准确率")
        axes[1].set_title("第四步：官方模型横向对比", fontsize=13)
        for b, a in zip(bars, m_acc):
            axes[1].text(a + 0.02, b.get_y() + b.get_height() / 2, f"{a:.1%}",
                         va="center", fontsize=11, fontweight="bold")
        axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("猫狗分类实验 · 结果汇总（同一份数据划分，种子 42）", fontsize=14)
    fig.tight_layout()
    out = OUTPUT_DIR / "experiment_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[生成] {out}")


def fig_independent_test():
    if not TEST_DIR.exists():
        print("[跳过] 没有 test_images/ 目录")
        return
    files = sorted(TEST_DIR.glob("*.jpg"))
    if not files:
        print("[跳过] test_images/ 里没有图片")
        return

    device = get_device()
    net = load_model(str(MODELS_DIR / "best_model.pth"), device)

    cols, rows = 5, (len(files) + 4) // 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.0))
    axes = axes.ravel()

    correct = 0
    for ax, p in zip(axes, files):
        truth = "cat" if p.name.lower().startswith("cat") else "dog"
        label, conf, _ = predict(p, net, device)
        ok = truth in label
        correct += ok
        ax.imshow(Image.open(p).convert("RGB"))
        ax.axis("off")
        cn = "猫" if truth == "cat" else "狗"
        ax.set_title(f"真实 {cn}\n预测 {label.split()[0]} {conf:.0%}",
                     fontsize=9.5, color="#1a7f37" if ok else "#c0392b")
    for ax in axes[len(files):]:
        ax.axis("off")

    fig.suptitle(f"独立测试集（与训练数据不重叠）　{correct}/{len(files)} 正确"
                 f" = {correct / len(files):.1%}　|　红色为误判", fontsize=13)
    fig.tight_layout()
    out = OUTPUT_DIR / "independent_test.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[生成] {out}  （{correct}/{len(files)}）")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_banner("生成实验图表")
    fig_experiment_summary()
    fig_independent_test()


if __name__ == "__main__":
    main()
