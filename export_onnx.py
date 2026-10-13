# -*- coding: utf-8 -*-
"""
拓展步骤：把训练好的模型导出为 ONNX，供网页端（onnxruntime-web）使用

为什么需要这一步？
    指导书第六步用 Gradio —— 它需要一台一直开着的电脑跑 Python 进程，
    链接是 http://127.0.0.1:7860，别人访问不到。
    如果想让别人点开一个网址就能用，就需要把模型搬到浏览器里跑：
    PyTorch 无法在浏览器运行 → 转成 ONNX → 用 onnxruntime-web（WASM）加载。
    这样整个网页是纯静态的，可以托管到任何静态服务器上。

预处理必须与 step5_test.py 完全一致：Resize((224,224)) + ToTensor + ImageNet 归一化

运行：python export_onnx.py
"""
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from common import CLASSES, MODELS_DIR, OUTPUT_DIR, ROOT

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
WEB_MODEL_DIR = ROOT / "web" / "model"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class ExportWrapper(nn.Module):
    """包一层 softmax，让 ONNX 的输出直接是概率，前端不用再算。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)


def load_net(weight_path):
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(str(weight_path), map_location="cpu"))
    return net.eval()


def collect_images(folder):
    folder = folder if hasattr(folder, "rglob") else __import__("pathlib").Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main():
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)

    ckpt = MODELS_DIR / "best_model.pth"
    if not ckpt.exists():
        print(f"[错误] 找不到 {ckpt}，请先运行 python step5_save_best.py")
        return 1

    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    net = load_net(ckpt)
    result = {"classes": list(CLASSES), "img_size": IMG_SIZE,
              "preprocess": "Resize((224,224)) -> ToTensor -> ImageNet Normalize"}

    # ---------------- 1) 导出 FP32 ONNX ---------------- #
    fp32_path = WEB_MODEL_DIR / "catdog_fp32.onnx"
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        ExportWrapper(net), dummy, str(fp32_path),
        input_names=["input"], output_names=["probabilities"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    fp32_mb = fp32_path.stat().st_size / 1024 / 1024
    print(f"[导出] FP32 ONNX -> {fp32_path}  ({fp32_mb:.1f} MB)")
    result["fp32"] = {"size_mb": round(fp32_mb, 2)}

    # ---------------- 2) 数值一致性校验 ---------------- #
    def torch_probs(p):
        # ⚠️ 必须每次都确保是 eval 模式：torch.onnx.export 结束后会把模型切回 train，
        # 而 BatchNorm 在 train 模式下会用"当前这一批"的统计量（这里 batch=1），
        # 结果完全失真 —— 会导致校验时 PyTorch 侧准确率莫名掉到一半。
        net.eval()
        with torch.no_grad():
            return torch.softmax(net(TF(Image.open(p).convert("RGB")).unsqueeze(0)), 1)[0].numpy()

    def onnx_probs(sess, p):
        x = TF(Image.open(p).convert("RGB")).unsqueeze(0).numpy()
        return np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0]).reshape(-1)

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    eval_dir = ROOT / "test_images"
    images = collect_images(eval_dir) if eval_dir.exists() else collect_images(ROOT / "datasets")[:40]

    max_diff, mismatch = 0.0, 0
    for p in images:
        a, b = torch_probs(p), onnx_probs(sess_fp32, p)
        max_diff = max(max_diff, float(np.abs(a - b).max()))
        mismatch += int(a.argmax() != b.argmax())
    print(f"[校验] PyTorch vs ONNX FP32：最大概率偏差 {max_diff:.2e}，"
          f"预测不一致 {mismatch}/{len(images)} 张")
    result["verification"] = {"max_prob_diff": max_diff, "mismatch": mismatch,
                              "num_images": len(images)}

    # ---------------- 3) 静态量化成 INT8 ---------------- #
    class PetCalibReader(CalibrationDataReader):
        def __init__(self, folder, limit=100):
            self.files = collect_images(folder)[:limit]
            self.it = iter(self.files)

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            return {"input": TF(Image.open(p).convert("RGB")).unsqueeze(0).numpy()}

    int8_path = WEB_MODEL_DIR / "catdog_int8.onnx"
    quantize_static(
        model_input=str(fp32_path), model_output=str(int8_path),
        calibration_data_reader=PetCalibReader(ROOT / "datasets"),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
        per_channel=True,
    )
    int8_mb = int8_path.stat().st_size / 1024 / 1024
    print(f"[量化] INT8 ONNX -> {int8_path}  ({int8_mb:.1f} MB，"
          f"压到 FP32 的 {int8_mb / fp32_mb:.0%})")

    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    acc = {"torch": 0, "fp32": 0, "int8": 0}
    n = 0
    for p in images:
        truth = 0 if p.name.lower().startswith(("cat", "猫")) else 1
        acc["torch"] += int(torch_probs(p).argmax() == truth)
        acc["fp32"] += int(onnx_probs(sess_fp32, p).argmax() == truth)
        acc["int8"] += int(onnx_probs(sess_int8, p).argmax() == truth)
        n += 1
    if n:
        print(f"[精度] {n} 张独立测试图：PyTorch {acc['torch']}/{n}  "
              f"ONNX-FP32 {acc['fp32']}/{n}  ONNX-INT8 {acc['int8']}/{n}")
    result["int8"] = {"size_mb": round(int8_mb, 2),
                      "accuracy": {k: f"{v}/{n}" for k, v in acc.items()}}

    # FP32 版体积大且精度与 INT8 相同，网页端只用 INT8，导出后即可删除
    fp32_path.unlink(missing_ok=True)
    print("[清理] 已删除网页用不到的 FP32 版本，网页只加载 INT8")

    (OUTPUT_DIR / "onnx_export_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[保存] 导出报告 -> {OUTPUT_DIR / 'onnx_export_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
