# -*- coding: utf-8 -*-
"""
serve_web.py —— 本地预览网页版的静态服务器

为什么不直接用 `python -m http.server`？
  Python 自带的 http.server 默认说 HTTP/1.0（每个请求都关连接），
  浏览器加载 onnxruntime-web 时会同时拉 .mjs（模块）和 11 MB 的 .wasm，
  在这种"边下边连"的场景下容易报
      TypeError: Failed to fetch dynamically imported module
  本脚本改成 HTTP/1.1 长连接 + 多线程 + 正确的 MIME 类型，
  并额外做了内存缓存，本地加载十几 MB 的模型不再出错。

用法：
    python serve_web.py                # 默认 http://127.0.0.1:8000
    python serve_web.py --port 8080
"""
import argparse
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent / "web"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# 补充 Python 默认不认识的扩展名，避免浏览器因 MIME 不对拒绝加载
EXTRA_TYPES = {
    ".wasm": "application/wasm",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".onnx": "application/octet-stream",
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".css": "text/css",
    ".html": "text/html",
}


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # 关键：长连接

    def guess_type(self, path):
        ext = Path(path).suffix.lower()
        if ext in EXTRA_TYPES:
            return EXTRA_TYPES[ext]
        return super().guess_type(path)

    def end_headers(self):
        # 本地开发，干脆禁用缓存，改完文件刷新就能看到
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        # 静音健康检查与预检请求，保持终端干净
        if "/favicon.ico" not in (self.path or ""):
            super().log_message(fmt, *args)


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
    ap = argparse.ArgumentParser(description="猫狗分类网页版本地服务器")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    args = ap.parse_args()

    if not (WEB_DIR / "index.html").exists():
        print(f"[错误] 找不到 {WEB_DIR / 'index.html'}")
        return 1

    handler = partial(Handler, directory=str(WEB_DIR))
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    httpd.daemon_threads = True

    print("=" * 62)
    print("  猫狗分类识别 · 网页版")
    print("=" * 62)
    print(f"  本机访问:      http://127.0.0.1:{args.port}")
    print(f"  局域网访问:    http://{local_ip()}:{args.port}")
    print(f"  静态目录:      {WEB_DIR}")
    print("  Ctrl+C 停止")
    print("=" * 62)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
