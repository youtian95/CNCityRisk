@echo off
chcp 65001 
echo 正在启动CNCityRisk项目...

REM 获取当前脚本所在目录
set SCRIPT_DIR=%~dp0

REM 检查是否存在 .env 环境目录
if exist "%SCRIPT_DIR%.env" (
    echo 找到本地conda环境: %SCRIPT_DIR%.env
    echo 激活本地环境...
    call conda activate "%SCRIPT_DIR%.env"
    if %errorlevel% neq 0 (
        echo 错误: 无法激活本地环境 %SCRIPT_DIR%.env
        echo 请检查环境是否正确创建
        pause
        exit /b 1
    )
    echo 本地环境激活成功
) else (
    echo 警告: 未找到本地环境 .env，使用默认conda环境
    call conda activate base
    if %errorlevel% neq 0 (
        echo 错误: 无法激活conda base环境
        echo 请确保conda已正确安装并配置
        pause
        exit /b 1
    )
)

REM 切换到脚本所在目录
cd /d "%SCRIPT_DIR%"

REM 显示当前Python环境信息
echo.
echo 当前Python环境信息:
python --version
echo 当前工作目录: %CD%

REM 运行Main.py
echo.
echo 正在运行Main.py...
python Main.py

REM 如果执行失败，显示错误信息
if %errorlevel% neq 0 (
    echo.
    echo 错误: Main.py执行失败，错误代码: %errorlevel%
)

echo.
echo 程序执行完成，按任意键退出...
pause