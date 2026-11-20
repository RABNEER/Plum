@echo off
cd /d "%~dp0"
python live_lab.py
if errorlevel 1 pause
