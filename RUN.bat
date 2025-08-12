@echo off
call venv\Scripts\activate.bat
python birefnet_cli.py %*
pause
