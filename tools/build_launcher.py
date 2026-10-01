# -*- coding: utf-8 -*-
"""构建托盘启动器 exe:  python tools/build_launcher.py"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
subprocess.run([sys.executable, "-m", "PyInstaller", "--onefile", "--noconsole",
                "--name", "启动英语全套学习", "--icon", os.path.join(ROOT, "launcher", "app.ico"),
                "--distpath", "dist/english-dictation",
                "--workpath", "build/launcher", "--specpath", "build/launcher",
                "launcher/launcher.py"], check=True)
print("完成: dist/english-dictation/启动英语全套学习.exe")
