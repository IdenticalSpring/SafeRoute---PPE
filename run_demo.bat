@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 demo.py
goto finish
:use_python
python demo.py
:finish
set "demo_exit=%errorlevel%"
echo.
pause
exit /b %demo_exit%
