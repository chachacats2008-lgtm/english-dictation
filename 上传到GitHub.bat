@echo off

chcp 65001 >nul

title 上传到 GitHub

cd /d %~dp0

echo ============================================

echo   英语全套学习 - 上传到 GitHub（送礼专用脚本）

echo ============================================

echo.

echo 前提：你已经在网页上建好了空仓库（见使用说明）。

echo.

set /p REPO=请粘贴你的仓库地址（形如 https://github.com/你的用户名/english-dictation.git）: 

if "%REPO%"=="" echo 没有输入地址 & pause & exit /b 1

git remote remove origin >nul 2>nul

git remote add origin %REPO%

echo.

echo 正在推送（第一次会弹出 GitHub 登录窗口，用浏览器登录即可）...

git push -u origin main

if errorlevel 1 (

  echo.

  echo 推送失败：请检查仓库地址是否正确、是否已在网页上创建仓库

) else (

  echo.

  echo ✅ 上传成功！仓库地址：%REPO%

)

echo.

pause

