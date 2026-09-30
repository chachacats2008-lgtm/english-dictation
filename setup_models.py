# -*- coding: utf-8 -*-
"""首次使用：下载本地语音与评测模型（约 400MB，一次性）。
国内优先走 hf-mirror 镜像；海外自动回退 huggingface 官方。"""
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")

VOICES = ["af_heart", "af_bella", "af_nicole", "am_adam",
          "am_michael", "am_puck", "bf_emma", "bm_george"]


def try_download(url, dest, min_size=1000):
    print("  下载:", url)
    try:
        urllib.request.urlretrieve(url, dest)
        if os.path.getsize(dest) < min_size:
            raise RuntimeError("文件过小")
        print("  完成:", dest, "(%.0f MB)" % (os.path.getsize(dest) / 1048576))
        return True
    except Exception as e:
        print("  失败:", e)
        if os.path.exists(dest):
            os.remove(dest)
        return False


def fetch(rel, dest, min_size=1000):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print("已存在，跳过:", os.path.relpath(dest, ROOT))
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for base in ("https://hf-mirror.com/", "https://huggingface.co/"):
        if try_download(base + rel, tmp, min_size):
            os.replace(tmp, dest)
            return True
    return False


def main():
    os.makedirs(MODELS, exist_ok=True)
    ok = True

    print("== 1/3 Kokoro 语音模型 (326MB) ==")
    ok &= fetch("onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx",
                os.path.join(MODELS, "model.onnx"), 100 * 1048576)

    print("== 2/3 音色文件 x%d ==" % len(VOICES))
    for v in VOICES:
        ok &= fetch("onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/%s.bin" % v,
                    os.path.join(MODELS, v + ".bin"), 400 * 1024)
    stub = os.path.join(MODELS, "voices.npy")
    if not os.path.exists(stub):
        open(stub, "wb").close()   # 空占位: 服务端优先读取单个 .bin 音色

    print("== 3/3 Vosk 发音评测模型 (40MB) ==")
    vdir = os.path.join(MODELS, "vosk-en", "vosk-model-small-en-us-0.15")
    if os.path.isdir(vdir):
        print("已存在，跳过: models/vosk-en")
    else:
        z = os.path.join(MODELS, "vosk.zip")
        got = try_download("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
                           z, 30 * 1048576)
        if not got:
            ok = False
        else:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(os.path.join(MODELS, "vosk-en"))
            os.remove(z)
            print("  解压完成")

    print()
    if ok:
        print("全部模型就绪！运行 python server.py 即可使用。")
    else:
        print("部分模型下载失败，请检查网络后重新运行本脚本。")
        sys.exit(1)


if __name__ == "__main__":
    main()
