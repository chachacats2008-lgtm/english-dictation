@echo off
chcp 65001 >nul
cd /d %~dp0
echo ================================================
echo   英语句子听写服务器启动中...
echo   Local: http://localhost:8111
echo   Phone: http://本机IP:8111  (手机与电脑同一WiFi)
echo ================================================
python server.py
pause
