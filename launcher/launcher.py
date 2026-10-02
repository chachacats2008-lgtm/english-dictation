# -*- coding: utf-8 -*-
"""英语全套学习 - 托盘启动器
双击后后台启动本地服务并自动打开浏览器, 缩到系统托盘;
托盘图标右键菜单: 打开学习页面 / 重启服务 / 退出。
单实例保护: 重复双击不会产生新实例, 而是直接打开学习页面。
"""
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

import pystray
from PIL import Image

PORT = 8111
GUARD_PORT = 8113          # 单实例哨兵端口
APP_DIR = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
           else os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(APP_DIR, "runtime", "python", "python.exe")
SERVER = os.path.join(APP_DIR, "server.py")
LOG = os.path.join(APP_DIR, "server-log.txt")

proc = None


def server_alive():
    try:
        urllib.request.urlopen("http://localhost:%d/api/health" % PORT, timeout=2)
        return True
    except Exception:
        return False


def _alert(msg):
    """无控制台模式下的错误提示框"""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, "英语全套学习", 0x10)
    except Exception:
        pass


def start_server():
    """启动本地服务(隐藏窗口, 日志写文件); 已在运行则不动"""
    global proc
    if server_alive():
        return
    if not (os.path.exists(RUNTIME) and os.path.exists(SERVER)):
        _alert("找不到 " + RUNTIME + " 或 " + SERVER + chr(10) +
               "请确认本程序位于应用文件夹内")
        sys.exit(1)
    logf = open(LOG, "ab")
    flags = 0x08000000  # CREATE_NO_WINDOW
    proc = subprocess.Popen([RUNTIME, "-u", SERVER], cwd=APP_DIR,
                            stdout=logf, stderr=logf, creationflags=flags)


def restart_server(icon=None, item=None):
    global proc
    if proc:
        try:
            proc.terminate()
        except Exception:
            pass
        proc = None
    time.sleep(1)
    if not server_alive():
        start_server()
    open_page()


def wait_ready(sec=90):
    for _ in range(sec * 2):
        if server_alive():
            return True
        time.sleep(0.5)
    return False


def open_page(icon=None, item=None):
    webbrowser.open("http://localhost:%d" % PORT)


def quit_app(icon, item):
    if proc:
        try:
            proc.terminate()
        except Exception:
            pass
    icon.stop()


def tray_image():
    p = os.path.join(APP_DIR, "icon-192.png")
    if os.path.exists(p):
        return Image.open(p)
    return Image.new("RGB", (64, 64), (67, 56, 202))


def acquire_primary():
    """单实例锁。已有实例在运行: 通知它打开页面并返回 False;
    本实例是第一个: 绑定哨兵端口监听后续通知, 返回 True。"""
    try:
        c = socket.create_connection(("127.0.0.1", GUARD_PORT), timeout=1)
        c.sendall(b"open")
        c.close()
        return False
    except OSError:
        pass
    try:
        srv = socket.socket()
        srv.bind(("127.0.0.1", GUARD_PORT))
        srv.listen(2)
    except OSError:
        webbrowser.open("http://localhost:%d" % PORT)   # 端口被占的兜底
        return False

    def watch():
        while True:
            try:
                c, _ = srv.accept()
                data = c.recv(16)
                c.close()
                if data:
                    open_page()
            except OSError:
                pass

    threading.Thread(target=watch, daemon=True).start()
    return True


def main():
    if not acquire_primary():
        sys.exit(0)               # 已有实例: 它会替你打开页面
    start_server()
    menu = pystray.Menu(
        pystray.MenuItem("📖 打开学习页面", open_page, default=True),
        pystray.MenuItem("🔄 重启服务", restart_server),
        pystray.MenuItem("❌ 退出（关闭服务）", quit_app),
    )
    icon = pystray.Icon("english_all_in_one", tray_image(),
                        "英语全套学习 - 服务运行中", menu)

    def opener():
        if wait_ready():
            webbrowser.open("http://localhost:%d" % PORT)
        else:
            _alert("服务启动超时，请查看应用文件夹里的 server-log.txt")

    threading.Thread(target=opener, daemon=True).start()
    icon.run()


if __name__ == "__main__":
    main()
