@echo off
rem 生成 HTTPS 自签名证书（iPhone 录音跟读需要）
where openssl >nul 2>nul
if errorlevel 1 (
  echo 未找到 openssl，请安装 Git for Windows 后重试（自带 openssl）
  pause
  exit /b 1
)
if not exist certs mkdir certs
set MSYS_NO_PATHCONV=1
openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem -days 3650 -nodes -subj "/CN=dictation-local"
echo.
echo 证书已生成到 certs\ 目录，重启 server.py 后可用 https://电脑IP:8112 访问
pause
