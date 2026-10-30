# 从绿色版生成公开体验版: 去掉外层目录、剔除新东方 l*.json 私有数据
# 用法: python tools/make_public_package.py
import os, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'dist', '英语全套学习-绿色版.zip')
DST = os.path.join(ROOT, 'dist', '英语全套学习-公开体验版.zip')
PREFIX = 'english-dictation/'

def main():
    tmp = DST + '.tmp'
    zin = zipfile.ZipFile(SRC)
    zout = zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED)
    kept = dropped = 0
    for it in zin.infolist():
        name = it.filename
        if not name.startswith(PREFIX):
            continue
        rel = name[len(PREFIX):]
        if not rel:
            continue
        base = os.path.basename(rel)
        if rel.startswith('data/') and base.startswith('l') and base.endswith('.json'):
            dropped += 1     # 新东方数据仅限私发绿色版
            continue
        it.filename = rel          # 去掉外层目录名
        zout.writestr(it, zin.read(name))
        kept += 1
    zout.close(); zin.close()
    os.replace(tmp, DST)
    print('完成: %s (%d 个文件, 剔除私有数据 %d 个)' % (DST, kept, dropped))

if __name__ == '__main__':
    main()
