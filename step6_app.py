# -*- coding: utf-8 -*-
"""
第六步：封装成可拍照分类的小程序（Gradio）

对应实验指导书第 6 节。部署三要素：
  加载训练好的权重 → 接收用户输入（拍照 / 上传）→ 返回预测结果

启动后：
  · 本机访问      http://127.0.0.1:7860
  · 手机同 Wi-Fi  http://<本机IP>:7860     ← 这就是课堂上的"拍照小程序"
  · 需要临时公网链接时：python step6_app.py --share

运行：python step6_app.py
"""
import argparse
import socket
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from common import CLASSES, MODELS_DIR, get_device

CLASSES_CN = {0: "猫", 1: "狗"}
PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

DEVICE = get_device()
WEIGHT = MODELS_DIR / "best_model.pth"


# ---------- 1. 加载模型（启动时只做一次） ----------
def load_net():
    if not WEIGHT.exists():
        raise FileNotFoundError(
            f"找不到权重文件 {WEIGHT}\n请先运行：python step5_save_best.py")
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(str(WEIGHT), map_location="cpu"))
    return net.to(DEVICE).eval()


NET = load_net()


# ---------- 2. 定义预测函数（每次拍照/上传都会调用） ----------
@torch.no_grad()
def predict(img: Image.Image):
    if img is None:
        return "请先拍照或上传一张图片", {}
    img = img.convert("RGB")
    x = PREDICT_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(NET(x), dim=1)[0].cpu()

    conf = float(probs.max())
    pred = int(probs.argmax())
    detail = {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)}

    tip = "" if conf >= 0.8 else "（置信度偏低，建议换个角度拍）"
    return f"识别结果：{CLASSES_CN[pred]}　置信度 {conf:.2%}　{tip}", detail


# ---------- 3. 搭建界面 ----------
def build_ui():
    with gr.Blocks(title="猫狗分类器") as demo:
        gr.Markdown(
            "## 猫狗分类器\n"
            "基于 **ResNet18 + ImageNet 迁移学习** 训练（100 张图片，验证集 80/20 划分）。\n\n"
            "对准猫或狗拍照，或上传一张图片即可识别。"
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", sources=["webcam", "upload"], label="拍照或上传")
                btn = gr.Button("开始识别", variant="primary")
            with gr.Column():
                out_text = gr.Textbox(label="识别结果", lines=2)
                out_label = gr.Label(label="各类别概率", num_top_classes=2)

        btn.click(predict, inputs=inp, outputs=[out_text, out_label])
        inp.change(predict, inputs=inp, outputs=[out_text, out_label])

        gr.Markdown(
            "---\n"
            "**提示**：若页面报错 `No module named 'gradio'`，先执行 `pip install gradio`。\n\n"
            "模型结构：手动搭建的 ResNet18 结构 + torchvision 预训练权重，"
            "只训练最后的全连接层。"
        )
    return demo


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description="猫狗分类 Gradio 应用")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="生成临时公网链接（gradio.live），需要联网")
    args = ap.parse_args()

    print(f"设备: {DEVICE}    权重: {WEIGHT}")
    demo = build_ui()
    print(f"\n本机访问:       http://127.0.0.1:{args.port}")
    print(f"手机同 Wi-Fi:   http://{local_ip()}:{args.port}\n")
    demo.launch(server_name="0.0.0.0", server_port=args.port,
                share=args.share, show_api=False)


if __name__ == "__main__":
    main()
