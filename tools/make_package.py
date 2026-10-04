# -*- coding: utf-8 -*-
"""把整个应用打包成绿色免安装包（朋友解压即用, 无需 Python/模型下载）
产物: dist/english-dictation/  +  dist/英语听写-绿色版.zip
"""
import os, shutil, subprocess, sys, zipfile, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, 'dist')
PKG = os.path.join(DIST, 'english-dictation')
PY_EMBED = 'https://mirrors.huaweicloud.com/python/3.12.7/python-3.12.7-embed-amd64.zip'
PIP_INDEX = 'https://mirrors.aliyun.com/pypi/simple/'

REQS = [
    'kokoro-onnx', 'edge-tts', 'pymupdf', 'rapidocr-onnxruntime',
    'pillow', 'audiotsm', 'imageio-ffmpeg',
    'opencc-python-reimplemented', 'vosk',
]   # opencv 由 rapidocr_onnxruntime 自带依赖拉取(避免 headless/完整版冲突导致解析卡死)

def main():
    if os.path.isdir(PKG):
        shutil.rmtree(PKG)
    os.makedirs(PKG)

    # 1) 应用文件
    for f in ['server.py', 'index.html', 'review.html', 'README.md',
              'manifest.webmanifest', 'icon-192.png', 'icon-512.png']:
        fp = os.path.join(ROOT, f)
        if os.path.exists(fp):
            shutil.copy2(fp, PKG)
    shutil.copytree(os.path.join(ROOT, 'data'), os.path.join(PKG, 'data'))
    shutil.copytree(os.path.join(ROOT, 'certs'), os.path.join(PKG, 'certs'))
    os.makedirs(os.path.join(PKG, 'exports'), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, 'exports', 'essay_workbook.pdf'),
                 os.path.join(PKG, 'exports', 'essay_workbook.pdf'))

    # 2) 模型（只要必需的：fp32 主模型 + 音色 + Vosk；排除量化模型/测试文件）
    src_models = os.path.join(ROOT, 'models')
    dst_models = os.path.join(PKG, 'models')
    os.makedirs(dst_models)
    shutil.copy2(os.path.join(src_models, 'model.onnx'), dst_models)   # 311MB
    for f in os.listdir(src_models):
        if f.endswith('.bin') or f == 'voices.npy':
            shutil.copy2(os.path.join(src_models, f), dst_models)
    shutil.copytree(os.path.join(src_models, 'vosk-en'),
                    os.path.join(dst_models, 'vosk-en'))

    # 3) 内嵌 Python 运行时
    rt = os.path.join(PKG, 'runtime')
    os.makedirs(rt)
    z = os.path.join(DIST, 'py_embed.zip')
    if not os.path.exists(z):
        print('下载内嵌 Python ...')
        urllib.request.urlretrieve(PY_EMBED, z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(os.path.join(rt, 'python'))
    # VC++ 运行库(msvcp140 等): onnxruntime/numpy 依赖, 干净系统没有会启动失败
    import glob as _glob
    for dll in _glob.glob(os.path.join(os.environ['SystemRoot'], 'System32', 'msvcp140*.dll'))              + _glob.glob(os.path.join(os.environ['SystemRoot'], 'System32', 'vcruntime140*.dll')):
        try:
            shutil.copy2(dll, os.path.join(rt, 'python', os.path.basename(dll)))
        except OSError:
            pass
    pth = os.path.join(rt, 'python', 'python312._pth')
    open(pth, 'w').write('python312.zip\n.\nLib\\site-packages\nimport site\n')

    # 4) 依赖库装进 runtime/python/Lib/site-packages（._pth 指向的路径）
    sp = os.path.join(rt, 'python', 'Lib', 'site-packages')
    print('安装依赖库（约300MB, 几分钟）...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    '--target', sp, '-i', PIP_INDEX] + REQS, check=True)

    # 5) 启动脚本 + 说明
    open(os.path.join(PKG, '启动英语全套学习.bat'), 'w', encoding='gbk').write(
        '@echo off\r\n'
        'cd /d %~dp0\r\n'
        'echo 正在启动英语全套学习服务（首次加载语音模型约需十几秒）...\r\n'
        'start "" http://localhost:8111\r\n'
        'runtime\\python.exe server.py\r\n'
        'pause\r\n')
    open(os.path.join(PKG, '使用说明.txt'), 'w', encoding='utf-8').write(
        '英语全套学习 · 绿色免安装版\n'
        '========================\n\n'
        '1. 把整个文件夹解压到任意位置（不要放U盘里直接运行）\n'
        '2. 双击「启动英语全套学习.exe」（推荐）\n'
        '   - 程序缩到屏幕右下角托盘（小图标可能收在 ^ 里），右键可「打开学习页面」或「退出」\n'
        '   - Windows 可能提示"已保护你的电脑"，点「更多信息」→「仍要运行」\n'
        '   - 也可以双击「启动英语全套学习.bat」，效果相同（有黑色窗口）\n'
        '3. 浏览器会自动打开 http://localhost:8111\n'
        '4. 手机使用（同一WiFi）：浏览器打开启动窗口里显示的 http://电脑IP:8111\n'
        '   - 要用录音跟读评测的 iPhone：改用 https://电脑IP:8112 并信任证书\n'
        '     （或先访问 http://电脑IP:8111/cert 安装证书后无警告）\n\n'
        '常见问题\n'
        '--------\n'
        '- 语音首次生成较慢（CPU 模型），生成后永久缓存\n'
        '- 「⚡ 提前生成语音」可提前批量生成当前页音频\n'
        '- 词库数据都在 data\\ 目录，备份拷走即可\n'
        '- 录音评分记录在 recordings\\ 目录\n\n'
        '本包已内置：Kokoro 本地 AI 语音模型、Vosk 发音评测模型、\n'
        '全部依赖库和 Python 运行时，无需联网下载任何东西。\n')

    # 6) 压缩
    print('压缩中 ...')
    out = shutil.make_archive(os.path.join(DIST, '英语全套学习-绿色版'), 'zip',
                              root_dir=DIST, base_dir='english-dictation')
    sz = os.path.getsize(out) // 1048576
    print('完成: %s (%d MB)' % (out, sz))

if __name__ == '__main__':
    main()
