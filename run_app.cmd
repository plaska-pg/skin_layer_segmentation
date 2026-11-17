@echo off
rem Double-click to start the app (Windows opens .ps1 files in Notepad and may block them, hence this wrapper)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_app.ps1"
if errorlevel 1 pause
