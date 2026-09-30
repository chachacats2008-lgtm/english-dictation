# -*- coding: utf-8 -*-
"""
英语听写小助手 · 单词+例句版 服务器（本地 AI 语音）
==================================================
- 网页：手机/电脑浏览器访问 http://<电脑IP>:8111
- 语音引擎：
    1) 本地 Kokoro-82M 神经网络语音模型（默认，完全离线，AI 真人级发音）
       模型放在 models/ 目录；合成结果缓存在 tts-cache/，同文本+音色+语速只合成一次
    2) 在线微软 Edge 神经网络语音（后备，需联网）
- /api/voices   返回可用引擎和音色
- /api/tts      合成语音（wav/mp3）
- /api/warm     后台预生成一课的语音（保存课程后自动调用）
- /api/lessons  词库课程保存/读取/删除（data/*.json）
依赖：python -m pip install edge-tts kokoro-onnx
启动：python server.py 或双击 run.bat
"""
import asyncio
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(ROOT, "tts-cache")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

HOST = "0.0.0.0"
PORT = 8111

# ---------------- 本地 Kokoro 引擎 ----------------
KOKORO_VOICES = [
    ("af_heart",   "Heart · 女声 温暖自然（推荐）"),
    ("af_bella",   "Bella · 女声 活泼"),
    ("af_nicole",  "Nicole · 女声 柔和"),
    ("am_adam",    "Adam · 男声 沉稳"),
    ("am_michael", "Michael · 男声 清晰"),
    ("am_puck",    "Puck · 男声 轻快"),
    ("bf_emma",    "Emma · 女声 英音"),
    ("bm_george",  "George · 男声 英音"),
]

# ---------------- 在线 Edge 引擎（后备） ----------------
EDGE_VOICES = [
    ("en-US-AriaNeural", "Aria · 在线 美音女声"),
    ("en-US-JennyNeural", "Jenny · 在线 美音女声"),
    ("en-US-AnaNeural", "Ana · 在线 美音童声"),
    ("en-GB-SoniaNeural", "Sonia · 在线 英音女声"),
]
try:
    import edge_tts
    EDGE_OK = True
except ImportError:
    EDGE_OK = False

_kokoro_state = {"obj": None, "err": ""}
_kokoro_lock = threading.Lock()
_infer_lock = threading.Lock()


def _init_kokoro():
    """在后台线程加载本地模型（首次约 2-5 秒）。"""
    with _kokoro_lock:
        if _kokoro_state["obj"] is not None or _kokoro_state["err"]:
            return _kokoro_state["obj"]
        try:
            import numpy as np
            from kokoro_onnx import Kokoro
            model_path = None
            for cand in ("model.onnx", "model_quantized.onnx", "model_fp16.onnx"):
                p = os.path.join(MODELS_DIR, cand)
                if os.path.exists(p):
                    model_path = p
                    break
            if not model_path:
                raise RuntimeError("models/ 目录没有找到 Kokoro 模型文件")
            # 构造函数需要一个可 np.load 的 voices 文件；实际音色用数组直接传入
            voices_npy = os.path.join(MODELS_DIR, "voices.npy")
            if not os.path.exists(voices_npy):
                np.save(voices_npy, np.array([], dtype=np.float32))
            k = Kokoro(model_path, voices_npy)
            voice_arrays = {}
            for vid, _ in KOKORO_VOICES:
                vpath = os.path.join(MODELS_DIR, vid + ".bin")
                if os.path.exists(vpath):
                    voice_arrays[vid] = np.fromfile(vpath, dtype=np.float32).reshape(510, 1, 256)
            if not voice_arrays:
                raise RuntimeError("models/ 目录没有找到音色文件(*.bin)")
            _kokoro_state["obj"] = (k, voice_arrays)
            print("[kokoro] loaded:", os.path.basename(model_path),
                  "| voices:", len(voice_arrays))
        except Exception as e:
            _kokoro_state["err"] = str(e)
            print("[kokoro] FAILED:", e)
        return _kokoro_state["obj"]


_FFMPEG_EXE = None


def _ffmpeg_exe():
    global _FFMPEG_EXE
    if _FFMPEG_EXE is None:
        import imageio_ffmpeg
        _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG_EXE


# ---------------- 发音评测（本地 Vosk 离线识别 + 词级比对） ----------------
VOSK_DIR = os.path.join(MODELS_DIR, "vosk-en", "vosk-model-small-en-us-0.15")
_vosk_model = None
_vosk_lock = threading.Lock()
REC_DIR = os.path.join(ROOT, "recordings")
os.makedirs(REC_DIR, exist_ok=True)


def vosk_available():
    return os.path.isdir(VOSK_DIR)


def _get_vosk():
    global _vosk_model
    if _vosk_model is None:
        from vosk import Model
        _vosk_model = Model(VOSK_DIR)
    return _vosk_model


def pron_score(target, recognized):
    """词级编辑距离比对。返回 (分数0-100, [{w,status}], 判定文案)"""
    import re
    def norm(s):
        s = s.lower().replace("’", "'")
        return re.sub(r"[^a-z' ]", " ", s).split()
    def lev1(a, b):                     # 是否仅差一个字母(小模型易把 boy 听成 by)
        if a == b: return False
        if abs(len(a) - len(b)) > 1: return False
        if len(a) == len(b):
            return sum(1 for x, y in zip(a, b) if x != y) <= 1
        i = j = diff = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]: i += 1; j += 1
            else:
                diff += 1
                if len(a) > len(b): i += 1
                else: j += 1
                if diff > 1: return False
        return True
    tw, rw = norm(target), norm(recognized)
    if not tw:
        return 0, [], "目标句为空"
    if not rw:
        return 0, [{"w": w, "status": "miss"} for w in tw], "没有识别到读音，请靠近麦克风再试"
    m, n = len(tw), len(rw)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = 0 if tw[i-1] == rw[j-1] else 1
            dp[i][j] = min(dp[i-1][j-1] + c, dp[i-1][j] + 1, dp[i][j-1] + 1)
    # 回溯标注每个目标词: ok / near(近似) / wrong(错读) / miss(漏读)
    detail, i, j = [], m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if tw[i-1] == rw[j-1] else 1):
            if tw[i-1] == rw[j-1]:
                st = "ok"
            elif lev1(tw[i-1], rw[j-1]):
                st = "near"
            else:
                st = "wrong"
            detail.append({"w": tw[i-1], "h": rw[j-1], "status": st})
            i, j = i-1, j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            detail.append({"w": tw[i-1], "status": "miss"})
            i -= 1
        else:
            j -= 1
    detail.reverse()
    ok = sum(1.0 if d["status"] == "ok" else 0.75 if d["status"] == "near" else 0
             for d in detail)
    score = int(round(ok / m * 100))
    if score >= 90: verdict = "很棒！发音准确 🌟"
    elif score >= 75: verdict = "不错，个别词再练练"
    elif score >= 60: verdict = "基本能听清，多跟读几遍"
    else: verdict = "先听原声再慢慢跟读吧"
    return score, detail, verdict


def pron_evaluate(target, audio_bytes):
    """audio: 浏览器录音(webm/wav 均可)。返回 dict。"""
    import subprocess
    import json as _json
    proc = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"],
        input=audio_bytes, capture_output=True, timeout=30)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("audio decode failed")
    with _vosk_lock:
        import vosk as _vosk
        rec = _vosk.KaldiRecognizer(_get_vosk(), 16000)
        rec.SetWords(True)
        chunks, step = proc.stdout, 8000
        texts = []
        for k in range(0, len(chunks), step):
            if rec.AcceptWaveform(chunks[k:k+step]):
                texts.append(_json.loads(rec.Result()).get("text", ""))
        texts.append(_json.loads(rec.FinalResult()).get("text", ""))
    recognized = " ".join(t for t in texts if t).strip()
    score, detail, verdict = pron_score(target, recognized)
    return {"recognized": recognized, "score": score, "detail": detail, "verdict": verdict}


def _trim_to_speech(x, sr, lead_ms=80, tail_ms=150):
    """裁掉首尾多余静音, 只保留语音前后少量余量。
    不依赖理论偏移量, 对拉伸算法的内部延迟免疫, 不会切到语音。"""
    import numpy as np
    mag = np.abs(x)
    thres = max(0.004, 0.02 * float(mag.max() if mag.size else 0))
    idx = np.where(mag > thres)[0]
    if len(idx) == 0:
        return x
    start = max(0, int(idx[0]) - int(sr * lead_ms / 1000))
    end = min(len(x), int(idx[-1]) + int(sr * tail_ms / 1000))
    return x[start:end]


def _time_stretch(audio, sr, speed):
    """变速不变调。优先 ffmpeg atempo（广播级, 句首双元音/瞬态不被打碎）;
    失败退回 audiotsm WSOLA（垫静音 + VAD 裁剪）; 再失败退回原速。"""
    import numpy as np
    # ---- 1) ffmpeg atempo ----
    try:
        import subprocess
        buf = io.BytesIO()
        w = wave.open(buf, "wb")
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        w.close()
        proc = subprocess.run(
            [_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
             "-f", "wav", "-i", "pipe:0",
             "-filter:a", "atempo=%.4f" % float(speed),
             "-f", "s16le", "-c:a", "pcm_s16le", "pipe:1"],
            input=buf.getvalue(), capture_output=True, timeout=60)
        if proc.returncode == 0 and len(proc.stdout) > 256:
            out = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768
            n = int(sr * 0.01)                    # 10ms 淡入淡出防爆音
            if len(out) > 2 * n:
                out = out.copy()
                out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
                out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
            return out
        raise RuntimeError(proc.stderr.decode("utf-8", "ignore")[:200])
    except Exception as e:
        print("[tts] atempo failed (%s), falling back to WSOLA" % e)
    # ---- 2) WSOLA 兜底 ----
    try:
        import audiotsm
        import audiotsm.io.array
        audio = audio.astype(np.float32)
        pad = int(sr * 0.3)                       # 两侧各 300ms 静音缓冲
        padded = np.concatenate([np.zeros(pad, dtype=np.float32), audio,
                                 np.zeros(pad, dtype=np.float32)])
        reader = audiotsm.io.array.ArrayReader(padded.reshape(1, -1))
        writer = audiotsm.io.array.ArrayWriter(1)
        audiotsm.wsola(1, speed=float(speed)).run(reader, writer)
        out = writer.data.reshape(-1).astype(np.float32)
        out = _trim_to_speech(out, sr)
        n = int(sr * 0.01)
        if len(out) > 2 * n:
            out = out.copy()
            out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
            out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
        return out
    except Exception as e:
        print("[tts] time-stretch failed (%s); keep normal speed" % e)
        return audio


def kokoro_wav(text, voice_id, speed):
    k, voice_arrays = _init_kokoro()
    if k is None:
        raise RuntimeError("kokoro unavailable: %s" % _kokoro_state["err"])
    if voice_id not in voice_arrays:
        voice_id = sorted(voice_arrays)[0]
    lang = "en-gb" if voice_id.startswith("b") else "en-us"
    speed = min(1.0, max(0.5, float(speed)))
    import numpy as np
    # 裸词补句号: 无终止标点时 Kokoro 把整词当急促语句处理, 容易吞音;
    # 补句号让模型给出完整的 utterance-final 收尾韵律
    synth_text = text.strip()
    if len(synth_text) > 2 and synth_text[-1] not in ".!?":
        synth_text += "."
    with _infer_lock:  # CPU 推理串行化，避免老 CPU 上争抢
        # 始终按正常语速合成（音质最佳）；慢速用电声学时间拉伸处理，
        # 避免 Kokoro 模型低速合成时的金属电音
        audio, sr = k.create(text=synth_text, voice=voice_arrays[voice_id], speed=1.0, lang=lang)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if speed < 0.95:
        audio = _time_stretch(audio, sr, speed)
    # 首部垫 500ms 静音: 播放设备/蓝牙链路冷启动会削掉音频流的开头,
    # 垫了静音后削掉的只是无声段, 正式发音完整
    audio = np.concatenate([np.zeros(int(sr * 0.5), dtype=np.float32), audio])
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    w.close()
    return buf.getvalue(), "audio/wav"


def edge_mp3(text, voice_id, speed):
    rate_pct = int(round((float(speed) - 1.0) * 100))
    rate_pct = max(-70, min(30, rate_pct))

    async def _synth():
        tts = edge_tts.Communicate(text, voice_id, rate="%+d%%" % rate_pct)
        buf = bytearray()
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        if not buf:
            raise RuntimeError("empty audio")
        return bytes(buf)

    return asyncio.run(_synth()), "audio/mpeg"


def tts_cache_key(engine, voice_id, speed, text):
    # p3: 音频首部垫 500ms 静音(防播放设备冷启动削头)
    # ts4: 变速用 ffmpeg atempo(修复句首词被 WSOLA 打碎)
    tag = ""
    if engine == "kokoro":
        tag = "p3|"
        if float(speed) < 0.95:
            tag += "ts4|"
    return hashlib.sha1((tag + "%s|%s|%s|%s" % (engine, voice_id, speed, text)).encode("utf-8")).hexdigest()


def tts_bytes(engine, voice_id, speed, text):
    """合成（或读缓存）。返回 (bytes, content_type)。"""
    key = tts_cache_key(engine, voice_id, speed, text)
    meta = {"kokoro": (".wav",), "edge": (".mp3",)}
    ext = meta[engine][0]
    path = os.path.join(CACHE_DIR, key + ext)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read(), ("audio/wav" if ext == ".wav" else "audio/mpeg")
    if engine == "kokoro":
        data, ctype = kokoro_wav(text, voice_id, speed)
    else:
        data, ctype = edge_mp3(text, voice_id, speed)
    tmp = path + ".tmp-%d" % threading.get_ident()
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return data, ctype


# ---------------- 后台预热队列 ----------------
warm_queue = []
warm_state = {"total": 0, "done": 0, "running": False, "lesson": ""}


WARM_QUEUE_FILE = os.path.join(CACHE_DIR, "queue.json")


def _save_warm_queue():
    try:
        with open(WARM_QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(warm_queue, f, ensure_ascii=False)
    except Exception:
        pass


def _load_warm_queue():
    """重启后恢复未完成的生成队列(已生成的自动跳过)"""
    if not os.path.exists(WARM_QUEUE_FILE):
        return
    try:
        with open(WARM_QUEUE_FILE, encoding="utf-8") as f:
            pending = json.load(f)
    except Exception:
        return
    resumed = 0
    for it in pending:
        try:
            engine, voice, speed, text = it[0], it[1], it[2], str(it[3])
            key = tts_cache_key(engine, voice, speed, text)
            ext = ".wav" if engine == "kokoro" else ".mp3"
            if not os.path.exists(os.path.join(CACHE_DIR, key + ext)):
                warm_queue.append((engine, voice, speed, text))
                resumed += 1
        except Exception:
            pass
    warm_state["total"] = len(warm_queue) + warm_state["done"]
    try:
        os.remove(WARM_QUEUE_FILE)
    except Exception:
        pass
    if resumed:
        print("[warm] 重启恢复: %d 条待生成(已完成的自动跳过)" % resumed)


def warm_worker():
    while True:
        if not warm_queue:
            warm_state["running"] = False
            time.sleep(0.5)
            continue
        warm_state["running"] = True
        engine, voice, speed, text = warm_queue.pop(0)
        _save_warm_queue()                    # 剩余队列实时落盘, 重启可续
        try:
            tts_bytes(engine, voice, speed, text)
        except Exception as e:
            print("[warm] failed:", text[:30], e)
        warm_state["done"] += 1


threading.Thread(target=warm_worker, daemon=True).start()


def enqueue_warm(items, engine, voice, speed):
    added = 0
    for it in items:
        texts = []
        word = (it.get("word") or "").strip()
        sent = (it.get("sent") or "").strip()
        if word:
            texts.append(word)
            for ch in word.lower():
                if "a" <= ch <= "z":
                    texts.append(ch)
        if sent:
            texts.append(sent)
        for t in dict.fromkeys(texts):  # 去重保序
            key = tts_cache_key(engine, voice, speed, t)
            ext = ".wav" if engine == "kokoro" else ".mp3"
            if not os.path.exists(os.path.join(CACHE_DIR, key + ext)):
                warm_queue.append((engine, voice, speed, t))
                added += 1
    warm_state["total"] = len(warm_queue) + warm_state["done"]
    _save_warm_queue()
    return added


# ---------------- 内置初中词库（示例，可用编辑器导入《漫画秒记》的词表） ----------------
SEEDS = [
 {"title": "七年级 · 核心词汇", "items": [
  {"word": "ability", "zh": "n. 能力；才能", "sent": "She has the ability to solve problems.", "sentZh": "她有解决问题的能力。"},
  {"word": "above", "zh": "prep. 在……上面", "sent": "The picture is above the blackboard.", "sentZh": "那幅画在黑板上方。"},
  {"word": "accept", "zh": "v. 接受", "sent": "I accept your invitation.", "sentZh": "我接受你的邀请。"},
  {"word": "achieve", "zh": "v. 实现；达到", "sent": "You can achieve your dream.", "sentZh": "你能实现你的梦想。"},
  {"word": "across", "zh": "prep. 穿过；横过", "sent": "They walked across the bridge.", "sentZh": "他们走过了那座桥。"},
  {"word": "advice", "zh": "n. 建议", "sent": "Let me give you some advice.", "sentZh": "让我给你一些建议。"},
  {"word": "afraid", "zh": "adj. 害怕的", "sent": "Don't be afraid of making mistakes.", "sentZh": "不要害怕犯错。"},
  {"word": "alone", "zh": "adj./adv. 独自（的）", "sent": "He finished the work alone.", "sentZh": "他独自完成了工作。"},
  {"word": "although", "zh": "conj. 虽然", "sent": "Although it was late, she kept working.", "sentZh": "虽然很晚了，她仍在工作。"},
  {"word": "angry", "zh": "adj. 生气的", "sent": "My father was angry with me.", "sentZh": "我父亲生我的气了。"},
  {"word": "another", "zh": "adj. 另一个", "sent": "Would you like another apple?", "sentZh": "你想再要一个苹果吗？"},
  {"word": "answer", "zh": "v./n. 回答", "sent": "Please answer my question.", "sentZh": "请回答我的问题。"},
  {"word": "arrive", "zh": "v. 到达", "sent": "We arrived at the station at six.", "sentZh": "我们六点到站。"},
  {"word": "asleep", "zh": "adj. 睡着的", "sent": "The baby is asleep.", "sentZh": "婴儿睡着了。"},
  {"word": "attention", "zh": "n. 注意", "sent": "Please pay attention to the teacher.", "sentZh": "请注意听老师讲课。"},
  {"word": "become", "zh": "v. 变成", "sent": "He wants to become a doctor.", "sentZh": "他想成为一名医生。"},
  {"word": "believe", "zh": "v. 相信", "sent": "I believe you can do it well.", "sentZh": "我相信你能做好。"},
  {"word": "belong", "zh": "v. 属于", "sent": "This book belongs to Mary.", "sentZh": "这本书是玛丽的。"},
  {"word": "borrow", "zh": "v. 借入", "sent": "May I borrow your pen?", "sentZh": "我可以借你的钢笔吗？"},
  {"word": "break", "zh": "v. 打破；打断", "sent": "Be careful not to break the glass.", "sentZh": "小心别打碎玻璃。"},
  {"word": "bright", "zh": "adj. 明亮的", "sent": "The classroom is bright and clean.", "sentZh": "教室明亮又干净。"},
  {"word": "busy", "zh": "adj. 忙碌的", "sent": "My mother is busy today.", "sentZh": "我妈妈今天很忙。"},
  {"word": "celebrate", "zh": "v. 庆祝", "sent": "We celebrate the Spring Festival together.", "sentZh": "我们一起庆祝春节。"},
  {"word": "cheap", "zh": "adj. 便宜的", "sent": "This shirt is cheap and nice.", "sentZh": "这件衬衫便宜又好。"},
  {"word": "clever", "zh": "adj. 聪明的", "sent": "What a clever boy!", "sentZh": "多么聪明的男孩！"},
  {"word": "comfortable", "zh": "adj. 舒适的", "sent": "The sofa is very comfortable.", "sentZh": "这个沙发非常舒适。"},
  {"word": "dangerous", "zh": "adj. 危险的", "sent": "It is dangerous to swim here.", "sentZh": "在这里游泳很危险。"},
  {"word": "delicious", "zh": "adj. 美味的", "sent": "The noodles taste delicious.", "sentZh": "这些面条尝起来很美味。"},
  {"word": "describe", "zh": "v. 描述", "sent": "Can you describe your school?", "sentZh": "你能描述一下你的学校吗？"},
  {"word": "different", "zh": "adj. 不同的", "sent": "We come from different cities.", "sentZh": "我们来自不同的城市。"},
  {"word": "enough", "zh": "adj./adv. 足够的", "sent": "We have enough time to read.", "sentZh": "我们有足够的时间读书。"},
  {"word": "expensive", "zh": "adj. 昂贵的", "sent": "The watch is too expensive.", "sentZh": "这块手表太贵了。"},
  {"word": "favourite", "zh": "adj. 最喜欢的", "sent": "Green is my favourite colour.", "sentZh": "绿色是我最喜欢的颜色。"},
  {"word": "foreign", "zh": "adj. 外国的", "sent": "He is learning a foreign language.", "sentZh": "他正在学一门外语。"},
  {"word": "hobby", "zh": "n. 爱好", "sent": "My hobby is playing the piano.", "sentZh": "我的爱好是弹钢琴。"},
  {"word": "important", "zh": "adj. 重要的", "sent": "English is very important.", "sentZh": "英语非常重要。"}
 ]},
 {"title": "八年级 · 核心词汇", "items": [
  {"word": "accident", "zh": "n. 事故", "sent": "There was a car accident yesterday.", "sentZh": "昨天发生了一起车祸。"},
  {"word": "afford", "zh": "v. 负担得起", "sent": "We can't afford a new car.", "sentZh": "我们买不起新车。"},
  {"word": "agreement", "zh": "n. 协议；同意", "sent": "They reached an agreement at last.", "sentZh": "他们最终达成了协议。"},
  {"word": "ancient", "zh": "adj. 古代的", "sent": "Xi'an is an ancient city.", "sentZh": "西安是一座古城。"},
  {"word": "appear", "zh": "v. 出现", "sent": "A rainbow appeared after the rain.", "sentZh": "雨后出现了一道彩虹。"},
  {"word": "argue", "zh": "v. 争论；争吵", "sent": "They never argue with each other.", "sentZh": "他们从不争吵。"},
  {"word": "attend", "zh": "v. 出席；参加", "sent": "All students attended the meeting.", "sentZh": "全体学生参加了会议。"},
  {"word": "audience", "zh": "n. 观众", "sent": "The audience clapped for five minutes.", "sentZh": "观众鼓掌了五分钟。"},
  {"word": "avoid", "zh": "v. 避免", "sent": "We should avoid making the same mistake.", "sentZh": "我们应该避免犯同样的错误。"},
  {"word": "balance", "zh": "n./v. 平衡", "sent": "You need a balanced diet.", "sentZh": "你需要均衡的饮食。"},
  {"word": "behave", "zh": "v. 表现", "sent": "The children behaved well at school.", "sentZh": "孩子们在学校表现很好。"},
  {"word": "benefit", "zh": "n./v. 好处；有益", "sent": "Exercise benefits our health.", "sentZh": "锻炼有益于我们的健康。"},
  {"word": "certain", "zh": "adj. 确定的", "sent": "I'm certain that he will come.", "sentZh": "我确定他会来。"},
  {"word": "chance", "zh": "n. 机会", "sent": "This is a good chance to learn.", "sentZh": "这是学习的好机会。"},
  {"word": "choice", "zh": "n. 选择", "sent": "You can make your own choice.", "sentZh": "你可以自己做选择。"},
  {"word": "communicate", "zh": "v. 交流", "sent": "We communicate with each other by email.", "sentZh": "我们通过电子邮件交流。"},
  {"word": "compare", "zh": "v. 比较", "sent": "Compare the two pictures and find the differences.", "sentZh": "比较这两幅图并找出不同。"},
  {"word": "complete", "zh": "v. 完成 adj. 完整的", "sent": "The workers completed the bridge.", "sentZh": "工人们完成了那座桥。"},
  {"word": "concentrate", "zh": "v. 集中注意力", "sent": "Please concentrate on your homework.", "sentZh": "请专心做作业。"},
  {"word": "confident", "zh": "adj. 自信的", "sent": "She is confident about the exam.", "sentZh": "她对考试很有信心。"},
  {"word": "consider", "zh": "v. 考虑", "sent": "I'm considering changing my job.", "sentZh": "我在考虑换工作。"},
  {"word": "continue", "zh": "v. 继续", "sent": "The rain continued all day.", "sentZh": "雨下了一整天。"},
  {"word": "conversation", "zh": "n. 谈话", "sent": "I had a long conversation with him.", "sentZh": "我和他进行了一次长谈。"},
  {"word": "correct", "zh": "adj. 正确的 v. 改正", "sent": "Your answer is correct.", "sentZh": "你的答案是正确的。"},
  {"word": "create", "zh": "v. 创造", "sent": "The artist created a beautiful painting.", "sentZh": "那位艺术家创作了一幅美丽的画。"},
  {"word": "culture", "zh": "n. 文化", "sent": "Let's learn about Chinese culture.", "sentZh": "让我们了解中国文化。"},
  {"word": "decision", "zh": "n. 决定", "sent": "It's hard to make a decision.", "sentZh": "做决定很难。"},
  {"word": "develop", "zh": "v. 发展", "sent": "The city is developing fast.", "sentZh": "这座城市发展很快。"},
  {"word": "difficulty", "zh": "n. 困难", "sent": "He finished it without difficulty.", "sentZh": "他毫无困难地完成了它。"},
  {"word": "discover", "zh": "v. 发现", "sent": "Columbus discovered America in 1492.", "sentZh": "哥伦布于1492年发现了美洲。"},
  {"word": "education", "zh": "n. 教育", "sent": "Education is important for everyone.", "sentZh": "教育对每个人都很重要。"},
  {"word": "encourage", "zh": "v. 鼓励", "sent": "My teacher often encourages me.", "sentZh": "我的老师经常鼓励我。"},
  {"word": "environment", "zh": "n. 环境", "sent": "We must protect the environment.", "sentZh": "我们必须保护环境。"},
  {"word": "especially", "zh": "adv. 尤其", "sent": "I like fruit, especially apples.", "sentZh": "我喜欢水果，尤其是苹果。"},
  {"word": "experience", "zh": "n. 经验；经历", "sent": "She has rich teaching experience.", "sentZh": "她有丰富的教学经验。"},
  {"word": "famous", "zh": "adj. 著名的", "sent": "The town is famous for its lake.", "sentZh": "这个小镇因它的湖而闻名。"}
 ]},
 {"title": "中考 · 高频词汇", "items": [
  {"word": "advantage", "zh": "n. 优势", "sent": "Living in the city has many advantages.", "sentZh": "住在城市有很多优势。"},
  {"word": "allow", "zh": "v. 允许", "sent": "My parents allow me to watch TV on weekends.", "sentZh": "我父母允许我周末看电视。"},
  {"word": "apologize", "zh": "v. 道歉", "sent": "You should apologize to her.", "sentZh": "你应该向她道歉。"},
  {"word": "available", "zh": "adj. 可获得的", "sent": "The ticket is available now.", "sentZh": "票现在可以买到了。"},
  {"word": "beauty", "zh": "n. 美；美丽", "sent": "The beauty of the mountain moved us.", "sentZh": "大山的美打动了我们。"},
  {"word": "calm", "zh": "adj. 平静的 v. 使平静", "sent": "Keep calm when you are in danger.", "sentZh": "遇到危险时要保持冷静。"},
  {"word": "challenge", "zh": "n./v. 挑战", "sent": "Learning English is a big challenge.", "sentZh": "学英语是一个很大的挑战。"},
  {"word": "character", "zh": "n. 性格；角色", "sent": "She has a strong character.", "sentZh": "她性格坚强。"},
  {"word": "compete", "zh": "v. 竞争", "sent": "The two teams compete for the prize.", "sentZh": "两支队伍为奖品而竞争。"},
  {"word": "contribute", "zh": "v. 贡献", "sent": "Everyone can contribute to society.", "sentZh": "每个人都能为社会做贡献。"},
  {"word": "curious", "zh": "adj. 好奇的", "sent": "Children are curious about everything.", "sentZh": "孩子们对一切都很好奇。"},
  {"word": "discuss", "zh": "v. 讨论", "sent": "Let's discuss the problem together.", "sentZh": "我们一起讨论这个问题吧。"},
  {"word": "effort", "zh": "n. 努力", "sent": "He passed the exam with great effort.", "sentZh": "通过巨大努力他通过了考试。"},
  {"word": "excellent", "zh": "adj. 优秀的", "sent": "She did an excellent job.", "sentZh": "她做得非常出色。"},
  {"word": "goal", "zh": "n. 目标", "sent": "Work hard to reach your goal.", "sentZh": "努力实现你的目标。"},
  {"word": "habit", "zh": "n. 习惯", "sent": "Reading is a good habit.", "sentZh": "读书是一个好习惯。"},
  {"word": "improve", "zh": "v. 提高；改进", "sent": "I want to improve my spoken English.", "sentZh": "我想提高我的英语口语。"},
  {"word": "influence", "zh": "n./v. 影响", "sent": "Teachers have a great influence on students.", "sentZh": "老师对学生有很大的影响。"},
  {"word": "knowledge", "zh": "n. 知识", "sent": "Knowledge is power.", "sentZh": "知识就是力量。"},
  {"word": "memory", "zh": "n. 记忆；回忆", "sent": "She has a good memory.", "sentZh": "她记忆力很好。"},
  {"word": "mention", "zh": "v. 提到", "sent": "He didn't mention the accident.", "sentZh": "他没有提到那次事故。"},
  {"word": "patient", "zh": "adj. 耐心的 n. 病人", "sent": "Please be patient with the children.", "sentZh": "请对孩子们耐心一点。"},
  {"word": "perform", "zh": "v. 表演；执行", "sent": "The students performed a short play.", "sentZh": "学生们表演了一个短剧。"},
  {"word": "practise", "zh": "v. 练习", "sent": "Practise speaking English every day.", "sentZh": "每天练习说英语。"},
  {"word": "prepare", "zh": "v. 准备", "sent": "She is preparing for the exam.", "sentZh": "她正在准备考试。"},
  {"word": "progress", "zh": "n./v. 进步", "sent": "You have made great progress.", "sentZh": "你已经取得了很大的进步。"},
  {"word": "proud", "zh": "adj. 自豪的", "sent": "We are proud of our country.", "sentZh": "我们为我们的国家感到自豪。"},
  {"word": "realize", "zh": "v. 意识到；实现", "sent": "I didn't realize my mistake.", "sentZh": "我没有意识到我的错误。"},
  {"word": "receive", "zh": "v. 收到", "sent": "I received a letter from my friend.", "sentZh": "我收到了朋友的一封信。"},
  {"word": "remember", "zh": "v. 记得", "sent": "Remember to lock the door.", "sentZh": "记得锁门。"},
  {"word": "reply", "zh": "v./n. 回复", "sent": "He replied to my email quickly.", "sentZh": "他很快回复了我的电子邮件。"},
  {"word": "require", "zh": "v. 需要；要求", "sent": "The job requires patience.", "sentZh": "这份工作需要耐心。"},
  {"word": "seem", "zh": "v. 似乎", "sent": "The question seems difficult.", "sentZh": "这个问题似乎很难。"},
  {"word": "support", "zh": "v./n. 支持", "sent": "My family always support me.", "sentZh": "我的家人总是支持我。"},
  {"word": "succeed", "zh": "v. 成功", "sent": "Work hard and you will succeed.", "sentZh": "努力学习，你就会成功。"}
 ]}
]

_id_lock = threading.Lock()
_id_counter = [0]


def new_lesson_id():
    with _id_lock:
        _id_counter[0] += 1
        return "l%d%d" % (int(time.time() * 1000), _id_counter[0])


def seed_if_empty():
    if any(f.endswith(".json") for f in os.listdir(DATA_DIR)):
        return
    for i, lesson in enumerate(SEEDS):
        lesson = dict(lesson)
        lesson["id"] = "seed%d" % (i + 1)
        lesson["created"] = i
        with open(os.path.join(DATA_DIR, lesson["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(lesson, f, ensure_ascii=False, indent=1)


def load_all_lessons():
    lessons = []
    for fn in os.listdir(DATA_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(DATA_DIR, fn), "r", encoding="utf-8") as f:
                lessons.append(json.load(f))
        except Exception:
            pass
    lessons.sort(key=lambda x: x.get("created", 0))
    return lessons


def lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def pdf_extract_text(data, max_pages=2000, max_chars=4000000):
    """从 PDF 提取文字层。返回 (总页数, 提取字符数, 文本)。"""
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    pages = doc.page_count
    texts = []
    total = 0
    for i in range(min(pages, max_pages)):
        try:
            t = (doc[i].get_text("text") or "").strip()
        except Exception:
            t = ""
        if t:
            texts.append("=== Page %d ===" % (i + 1))
            texts.append(t)
            total += len(t)
            if total > max_chars:
                break
    doc.close()
    return pages, total, "\n".join(texts)


import secrets as _secrets
import threading as _threading
PDF_TMP_DIR = os.path.join(ROOT, "pdf-tmp")
os.makedirs(PDF_TMP_DIR, exist_ok=True)
for _old in os.listdir(PDF_TMP_DIR):        # 启动时清掉上次残留的临时 PDF
    try:
        os.remove(os.path.join(PDF_TMP_DIR, _old))
    except OSError:
        pass
_pdf_cache = {}   # uploadId -> {"pages": {n: text}, "order": [n,...], "ts": float}
_pdf_files = {}   # uploadId -> {"path": filepath, "pages": int, "ts": float}


def _prune_pdf_maps():
    """两个缓存只各留最近 3 份，并删除被挤掉 pdf 的临时文件"""
    while len(_pdf_files) > 3:
        old_uid = min(_pdf_files, key=lambda k: _pdf_files[k]["ts"])
        ent = _pdf_files.pop(old_uid)
        try:
            os.remove(ent["path"])
        except OSError:
            pass
        _pdf_cache.pop(old_uid, None)
    while len(_pdf_cache) > 3:
        old_uid = min(_pdf_cache, key=lambda k: _pdf_cache[k]["ts"])
        _pdf_cache.pop(old_uid, None)
        ent = _pdf_files.pop(old_uid, None)
        if ent:
            try:
                os.remove(ent["path"])
            except OSError:
                pass


def _cache_pdf_file(uid, data, pages):
    fp = os.path.join(PDF_TMP_DIR, uid + ".pdf")
    with open(fp, "wb") as f:
        f.write(data)
    _pdf_files[uid] = {"path": fp, "pages": pages, "ts": time.time()}
    _prune_pdf_maps()


def _cache_pdf_text(full_text, uid=None):
    if uid is None:
        uid = _secrets.token_hex(8)
    pages = {}
    order = []
    import re as _re
    parts = _re.split(r"=== Page (\d+) ===", full_text)
    # parts: ['', '1', content1, '2', content2, ...]
    for i in range(1, len(parts) - 1, 2):
        try:
            n = int(parts[i])
        except ValueError:
            continue
        pages[n] = parts[i + 1].strip("\n")
        order.append(n)
    _pdf_cache[uid] = {"pages": pages, "order": order, "ts": time.time()}
    _prune_pdf_maps()
    return uid


def pdf_text_range(uid, page_from, page_to):
    ent = _pdf_cache.get(uid)
    if not ent:
        return None
    out = []
    for n in ent["order"]:
        if page_from <= n <= page_to:
            out.append("=== Page %d ===\n%s" % (n, ent["pages"][n]))
    return "\n".join(out)


_ocr_engine = None
_ocr_lock = _threading.Lock()


def ocr_available():
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def _get_ocr():
    """惰性加载 RapidOCR（首次调用要加载神经网络模型，约几秒）"""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def pdf_ocr_range(uid, page_from, page_to):
    """把 PDF 指定页渲染成图片后用神经网络 OCR。结果回填文字缓存。
    返回 (文本, 实际识别页数) ；uid 不存在返回 None。"""
    ent = _pdf_files.get(uid)
    if not ent:
        return None
    page_from = max(1, page_from)
    page_to = min(page_to, ent["pages"])
    if page_to < page_from:
        return "", 0
    import fitz
    import numpy as np
    ocr = _get_ocr()
    out = []
    done = 0
    with _ocr_lock:                       # CPU 推理串行, 避免并发把老机器拖死
        doc = fitz.open(ent["path"])
        try:
            for n in range(page_from, page_to + 1):
                try:
                    pix = doc[n - 1].get_pixmap(dpi=200)
                    img = np.frombuffer(pix.samples, dtype=np.uint8) \
                            .reshape(pix.height, pix.width, 3)[:, :, ::-1]   # RGB -> BGR
                    result, _ = ocr(img)
                except Exception as e:
                    print("[ocr] page %d failed: %s" % (n, e))
                    result = None
                lines = []
                if result:
                    for row in result:
                        try:
                            box, txt, score = row[0], row[1], row[2]
                        except Exception:
                            continue
                        txt = (txt or "").strip()
                        if not txt or (isinstance(score, (int, float)) and score < 0.45):
                            continue
                        ys = [p[1] for p in box]
                        xs = [p[0] for p in box]
                        lines.append((min(ys), min(xs), txt))
                lines.sort()               # 按从上到下、从左到右还原阅读顺序
                page_text = "\n".join(t for _, _, t in lines)
                out.append("=== Page %d ===\n%s" % (n, page_text))
                done += 1
                # 回填文字缓存: 之后 /api/pdf/text 也能直接取到, 不用重复识别
                ce = _pdf_cache.get(uid)
                if ce is not None:
                    if n not in ce["pages"]:
                        ce["order"].append(n)
                    ce["pages"][n] = page_text
        finally:
            doc.close()
    return "\n".join(out), done


def kokoro_available():
    have_model = any(os.path.exists(os.path.join(MODELS_DIR, f))
                     for f in ("model.onnx", "model_quantized.onnx", "model_fp16.onnx"))
    return have_model


class Handler(BaseHTTPRequestHandler):
    server_version = "DictServer/3"

    def log_message(self, fmt, *args):
        pass

    def _bytes(self, code, body, ctype, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._bytes(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        path = parsed.path

        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(ROOT, "index.html"), "rb") as f:
                    self._bytes(200, f.read(), "text/html; charset=utf-8",
                                {"Cache-Control": "no-cache"})
            except FileNotFoundError:
                self._json(404, {"error": "index.html not found"})
            return

        if path in ("/manifest.webmanifest", "/icon-192.png", "/icon-512.png"):
            fn = path.lstrip("/")
            mime = {"manifest.webmanifest": "application/manifest+json",
                    "icon-192.png": "image/png", "icon-512.png": "image/png"}[fn]
            try:
                with open(os.path.join(ROOT, fn), "rb") as f:
                    self._bytes(200, f.read(), mime, {"Cache-Control": "no-cache"})
            except FileNotFoundError:
                self._json(404, {"error": "not found"})
            return

        if path in ("/review", "/review.html"):
            try:
                with open(os.path.join(ROOT, "review.html"), "rb") as f:
                    self._bytes(200, f.read(), "text/html; charset=utf-8",
                                {"Cache-Control": "no-cache"})
            except FileNotFoundError:
                self._json(404, {"error": "review.html not found"})
            return

        if path == "/api/export/pdf":
            fp = os.path.join(ROOT, "exports", "essay_workbook.pdf")
            if not os.path.exists(fp):
                self._json(404, {"error": "PDF 尚未生成（exports/essay_workbook.pdf）"})
                return
            with open(fp, "rb") as f:
                data = f.read()
            fn = urllib.parse.quote("中考短文60篇·背诵听写本.pdf")
            self._bytes(200, data, "application/pdf", {
                "Content-Disposition":
                    "attachment; filename=\"essay_workbook.pdf\"; filename*=UTF-8''" + fn,
            })
            return

        if path == "/cert":
            # 手机安装信任证书入口: Safari 打开 http://<IP>:8111/cert 即触发描述文件下载
            try:
                with open(os.path.join(ROOT, "certs", "cert.pem"), "rb") as f:
                    self._bytes(200, f.read(), "application/x-x509-ca-cert",
                                {"Content-Disposition": "attachment; filename=dictation-cert.pem"})
            except FileNotFoundError:
                self._json(404, {"error": "cert not found"})
            return

        if path == "/api/pron/status":
            self._json(200, {"ok": True, "ready": vosk_available()})
            return

        if path == "/api/health":
            self._json(200, {
                "ok": True,
                "lan": "%s:%d" % (lan_ip(), PORT),
                "kokoro": kokoro_available(),
                "edge": EDGE_OK,
                "ocr": ocr_available(),
            })
            return

        if path == "/api/voices":
            engines = []
            if kokoro_available():
                engines.append({
                    "id": "kokoro", "name": "本地 AI 模型 Kokoro（离线，推荐）",
                    "voices": [["kokoro:" + v, n] for v, n in KOKORO_VOICES],
                })
            if EDGE_OK:
                engines.append({
                    "id": "edge", "name": "在线 微软语音（需联网）",
                    "voices": [["edge:" + v, n] for v, n in EDGE_VOICES],
                })
            self._json(200, {"engines": engines})
            return

        if path == "/api/warm/status":
            self._json(200, {
                "queued": len(warm_queue),
                "running": warm_state["running"],
                "done": warm_state["done"],
                "total": warm_state["total"],
            })
            return

        if path == "/api/pdf/ocr":
            uid = (q.get("uploadId") or [""])[0]
            try:
                pf = int((q.get("from") or ["1"])[0])
                pt = int((q.get("to") or ["1"])[0])
            except ValueError:
                self._json(400, {"error": "bad page range"})
                return
            if pf > pt:
                pf, pt = pt, pf
            ent = _pdf_files.get(uid)
            if not ent:
                self._json(404, {"error": "上传已过期，请重新上传 PDF"})
                return
            if pt - pf + 1 > 20:
                self._json(400, {"error": "AI 识别每次最多 20 页，请分批（如 1–20、21–40）"})
                return
            if not ocr_available():
                self._json(500, {"error": "OCR 模块未安装（pip install rapidocr_onnxruntime）"})
                return
            t0 = time.time()
            try:
                text, done = pdf_ocr_range(uid, pf, pt)
            except Exception as e:
                self._json(500, {"error": "OCR failed: %s" % e})
                return
            print("[ocr] pages %d-%d done in %.1fs" % (pf, pt, time.time() - t0))
            self._json(200, {"ok": True, "from": pf, "to": pt, "pages": done,
                             "text": text, "secs": round(time.time() - t0, 1)})
            return

        if path == "/api/pdf/text":
            uid = (q.get("uploadId") or [""])[0]
            try:
                pf = int((q.get("from") or ["1"])[0])
                pt = int((q.get("to") or ["999999"])[0])
            except ValueError:
                self._json(400, {"error": "bad page range"})
                return
            if pf > pt:
                pf, pt = pt, pf
            text = pdf_text_range(uid, pf, pt)
            if text is None:
                self._json(404, {"error": "上传已过期，请重新上传 PDF"})
                return
            if not text:
                self._json(200, {"ok": True, "from": pf, "to": pt, "text": "", "note": "该页码范围内没有文字内容"})
                return
            self._json(200, {"ok": True, "from": pf, "to": pt, "text": text})
            return

        if path == "/api/lessons":
            self._json(200, load_all_lessons())
            return

        if path == "/api/tts":
            text = (q.get("text") or [""])[0].strip()
            voice = (q.get("voice") or [""])[0].strip()
            try:
                speed = float((q.get("speed") or ["0.8"])[0])
            except ValueError:
                speed = 0.8
            if not text:
                self._json(400, {"error": "text is required"})
                return
            engine, _, voice_id = voice.partition(":")
            if engine not in ("kokoro", "edge"):
                engine = "kokoro" if kokoro_available() else "edge"
                voice_id = voice
            if engine == "kokoro" and not kokoro_available():
                engine = "edge"
                voice_id = EDGE_VOICES[0][0]
            if engine == "edge":
                valid = {v for v, _ in EDGE_VOICES}
                if voice_id not in valid:
                    voice_id = "en-US-AriaNeural"
                if not EDGE_OK:
                    self._json(502, {"error": "edge-tts not installed"})
                    return
            if len(text) > 400:
                text = text[:400]
            try:
                data, ctype = tts_bytes(engine, voice_id, speed, text)
                etag = '"%s"' % tts_cache_key(engine, voice_id, speed, text)
                inm = self.headers.get("If-None-Match") or ""
                if etag in [t.strip() for t in inm.split(",") if t.strip()]:
                    self.send_response(304)
                    self.send_header("ETag", etag)
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    return
                self._bytes(200, data, ctype,
                            {"Cache-Control": "no-cache", "ETag": etag})
            except Exception as e:
                print("[tts] FAILED engine=%s voice=%s err=%r" % (engine, voice_id, e))
                self._json(502, {"error": "tts failed: %s" % e})
            return

        self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        # 大文件 PDF 上传（原始请求体即 PDF 内容）
        if parsed.path == "/api/pdf/extract":
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length < 1000:
                self._json(400, {"error": "empty upload"})
                return
            if length > 900000000:
                self._json(413, {"error": "file too large (max ~900MB)"})
                return
            data = self.rfile.read(length)
            try:
                pages, chars, text = pdf_extract_text(data)
            except Exception as e:
                self._json(400, {"error": "pdf parse failed: %s" % e})
                return
            has_text = chars > 50
            print("[pdf] pages=%d textChars=%d hasText=%s" % (pages, chars, has_text))
            upload_id = _secrets.token_hex(8)
            _cache_pdf_file(upload_id, data, pages)
            preview = ""
            if has_text:
                _cache_pdf_text(text, uid=upload_id)
                preview = text[:300]
            self._json(200, {
                "ok": True,
                "uploadId": upload_id,
                "pages": pages,
                "textChars": chars,
                "hasText": has_text,
                "preview": preview,
                "note": "" if has_text else
                        "未检测到文字层：这是扫描图片版 PDF，请选择页码范围用 AI 识别（每次最多 20 页）",
            })
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            self._json(400, {"error": "bad json"})
            return

        if parsed.path == "/api/pron":
            text = str(body.get("text") or "").strip()
            audio_b64 = str(body.get("audio") or "")
            if not text or not audio_b64:
                self._json(400, {"error": "text and audio are required"})
                return
            if not vosk_available():
                self._json(503, {"error": "发音评测模型未就绪（models/vosk-en）"})
                return
            import base64 as _b64
            try:
                audio = _b64.b64decode(audio_b64)
            except Exception:
                self._json(400, {"error": "bad audio data"})
                return
            try:
                t0 = time.time()
                result = pron_evaluate(text, audio)
            except Exception as e:
                print("[pron] FAILED err=%r" % e)
                self._json(500, {"error": "评测失败: %s" % e})
                return
            # 采集: 录音与成绩一起存档
            import hashlib as _h
            fn = "%d_%s.webm" % (int(time.time() * 1000),
                                 _h.sha1(text.encode("utf-8")).hexdigest()[:8])
            try:
                with open(os.path.join(REC_DIR, fn), "wb") as f:
                    f.write(audio)
                with open(os.path.join(REC_DIR, fn + ".json"), "w", encoding="utf-8") as f:
                    json.dump({"text": text, "time": int(time.time()), **result},
                              f, ensure_ascii=False)
            except Exception as e:
                print("[pron] save failed:", e)
            result["secs"] = round(time.time() - t0, 1)
            self._json(200, {"ok": True, **result})
            return

        if parsed.path == "/api/lessons/save":
            title = str(body.get("title") or "").strip()[:60] or "未命名词库"
            kind = str(body.get("kind") or "").strip()
            if kind not in ("vocab", "sent", "mixed"):
                kind = "mixed"
            raw_items = body.get("items") or []
            items = []
            for it in raw_items[:800]:
                it = it or {}
                words = []
                for w in (it.get("words") or [])[:10]:
                    w = str(w).strip()[:60]
                    if w:
                        words.append(w)
                items.append({
                    "word": str(it.get("word") or "").strip()[:100],
                    "zh": str(it.get("zh") or "").strip()[:120],
                    "sent": str(it.get("sent") or "").strip()[:400],
                    "sentZh": str(it.get("sentZh") or "").strip()[:200],
                    "words": words,
                })
            items = [it for it in items if it["word"] or it["sent"]]
            if not items:
                self._json(400, {"error": "no valid entries"})
                return
            lid = str(body.get("id") or "").strip()
            if lid and not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", lid):
                lid = ""
            fp = os.path.join(DATA_DIR, lid + ".json") if lid else None
            if lid and fp and os.path.exists(fp):
                try:
                    with open(fp, encoding="utf-8") as f:
                        old = json.load(f)
                        created = old.get("created", int(time.time() * 1000))
                        if not body.get("kind"):  # 未指定则沿用原类型
                            kind = old.get("kind", "mixed")
                except Exception:
                    created = int(time.time() * 1000)
            else:
                lid = new_lesson_id()
                created = int(time.time() * 1000)
            lesson = {"id": lid, "title": title, "kind": kind, "items": items, "created": created}
            with open(os.path.join(DATA_DIR, lid + ".json"), "w", encoding="utf-8") as f:
                json.dump(lesson, f, ensure_ascii=False, indent=1)
            print("[lesson] saved id=%s title=%s items=%d" % (lid, title, len(items)))
            # 自动后台预热语音（用默认引擎/音色/常用慢速）
            try:
                engine, voice = ("kokoro", "af_heart") if kokoro_available() else ("edge", "en-US-AriaNeural")
                added = enqueue_warm(items, engine, voice, 0.65)
                print("[warm] enqueued %d" % added)
            except Exception as e:
                print("[warm] error", e)
            self._json(200, {"ok": True, "id": lid, "title": title, "items": len(items)})
            return

        if parsed.path == "/api/tts/regenerate":
            items = body.get("items") or []
            engine, _, voice = str(body.get("voice") or "").partition(":")
            if engine not in ("kokoro", "edge"):
                engine, voice = "kokoro", "af_heart"
            try:
                speed = float(body.get("speed") or 0.65)
            except (TypeError, ValueError):
                speed = 0.65
            # 按与预热相同的文本集(词+字母拼写+例句)删除缓存文件
            cleared = 0
            for it in items:
                it = it or {}
                texts = []
                word = (it.get("word") or "").strip()
                sent = (it.get("sent") or "").strip()
                if word:
                    texts.append(word)
                    for ch in word.lower():
                        if "a" <= ch <= "z":
                            texts.append(ch)
                if sent:
                    texts.append(sent)
                for t in dict.fromkeys(texts):
                    key = tts_cache_key(engine, voice, speed, t)
                    for ext in (".wav", ".mp3"):
                        fp = os.path.join(CACHE_DIR, key + ext)
                        if os.path.exists(fp):
                            try:
                                os.remove(fp)
                                cleared += 1
                            except OSError:
                                pass
            added = enqueue_warm(items, engine, voice, speed)
            print("[tts] regenerate cleared=%d requeued=%d" % (cleared, added))
            self._json(200, {"ok": True, "cleared": cleared, "queued": added})
            return

        if parsed.path == "/api/warm":
            items = body.get("items") or []
            engine, _, voice = str(body.get("voice") or "").partition(":")
            if engine not in ("kokoro", "edge"):
                engine, voice = ("kokoro", "af_heart") if kokoro_available() else ("edge", "en-US-AriaNeural")
            try:
                speed = float(body.get("speed") or 0.65)
            except (TypeError, ValueError):
                speed = 0.65
            added = enqueue_warm(items, engine, voice, speed)
            self._json(200, {"queued": added})
            return

        if parsed.path == "/api/lessons/delete":
            lid = str(body.get("id") or "").strip()
            if re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", lid):
                p = os.path.join(DATA_DIR, lid + ".json")
                if os.path.exists(p):
                    os.remove(p)
                    print("[lesson] deleted id=%s" % lid)
            self._json(200, {"ok": True})
            return

        self._json(404, {"error": "not found"})


def main():
    seed_if_empty()
    _load_warm_queue()
    # 后台预加载本地模型，首次请求更快
    if kokoro_available():
        threading.Thread(target=_init_kokoro, daemon=True).start()
    # HTTPS 端口（自签名证书）: iOS/Safari 的麦克风只允许安全上下文
    HTTPS_PORT = 8112
    if os.path.exists("certs/cert.pem") and os.path.exists("certs/key.pem"):
        try:
            import ssl
            hs = ThreadingHTTPServer(("0.0.0.0", HTTPS_PORT), Handler)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain("certs/cert.pem", "certs/key.pem")
            hs.socket = ctx.wrap_socket(hs.socket, server_side=True)
            threading.Thread(target=hs.serve_forever, daemon=True).start()
            print("[https] listening on https://<PC-IP>:%d (self-signed)" % HTTPS_PORT)
        except Exception as e:
            print("[https] disabled:", e)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print("=" * 56)
    print("  English Dictation Server v3 (words + sentences)")
    print("  Local :  http://localhost:%d" % PORT)
    print("  Phone :  http://<PC-IP>:%d   (same WiFi)" % PORT)
    print("  iOS麦克: https://<PC-IP>:%d  (录音需要, 信任证书后使用)" % HTTPS_PORT)
    print("  Engine:  Kokoro local AI = %s | Edge online = %s"
          % (kokoro_available(), EDGE_OK))
    print("=" * 56)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
