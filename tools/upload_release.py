# -*- coding: utf-8 -*-
"""把公开体验版 zip 上传到 GitHub Release（带重试，供弱网环境后台跑）
用法: GH_TOKEN=xxx python tools/upload_release.py dist/英语全套学习-公开体验版.zip
"""
import json
import os
import sys
import time
import urllib.request

TOKEN = os.environ.get("GH_TOKEN", "")
PROXY = "http://127.0.0.1:7890"
RELEASE_ID = 401428526
REPO = "chachacats2008-lgtm/english-dictation"
ASSET_NAME = "english-all-in-one-v1.0.zip"

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))


def api(url, method="GET", data=None, ctype="application/json", timeout=40):
    req = urllib.request.Request(url, method=method, data=data)
    req.add_header("Authorization", "Bearer " + TOKEN)
    req.add_header("Accept", "application/vnd.github+json")
    if data is not None:
        req.add_header("Content-Type", ctype)
    with opener.open(req, timeout=timeout) as r:
        return r.status, r.read()


def retry(fn, tries=8, wait=8, label=""):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            print("[%s] 第%d次失败: %s, %d秒后重试" % (label, i + 1, e, wait), flush=True)
            time.sleep(wait)
    raise RuntimeError(label + " 重试耗尽")


def main():
    zpath = sys.argv[1]
    data = open(zpath, "rb").read()
    print("开始: %.0fMB" % (len(data) / 1048576), flush=True)

    # 1) 删除同名旧资产(如有)
    def del_old():
        st, body = api("https://api.github.com/repos/%s/releases/%d/assets"
                       % (REPO, RELEASE_ID))
        for a in json.loads(body):
            if a["name"] == ASSET_NAME:
                api("https://api.github.com/repos/%s/releases/assets/%d"
                    % (REPO, a["id"]), method="DELETE")
                print("旧资产已删除", flush=True)
                return
        print("无旧资产", flush=True)
    retry(del_old, label="删除旧资产")

    # 2) 上传(大文件, 单次长超时, 失败重试)
    for attempt in range(3):
        try:
            print("上传第%d次..." % (attempt + 1), flush=True)
            st, body = api(
                "https://uploads.github.com/repos/%s/releases/%d/assets?name=%s"
                % (REPO, RELEASE_ID, ASSET_NAME),
                method="POST", data=data, ctype="application/zip", timeout=3600)
            info = json.loads(body)
            print("上传成功:", info.get("state"), info.get("browser_download_url"), flush=True)
            return
        except Exception as e:
            print("上传失败: %s" % e, flush=True)
            time.sleep(15)
            # 清理可能出现的半成品
            try:
                retry(del_old, tries=2, label="清理半成品")
            except Exception:
                pass
    raise RuntimeError("三次上传均失败")


if __name__ == "__main__":
    main()
