@echo off
title GOAT Studio
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo GOAT Studio is not installed yet. Running the installer first...
    call "Install GOAT Studio.bat"
)
rem Make sure the local AI server is running (Ollama starts minimized in the tray).
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if exist "%OLLAMA%" (
    tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul || start "" /min "%OLLAMA%" serve
)
echo Starting GOAT Studio at http://127.0.0.1:7860 ...
".venv\Scripts\python.exe" app.py
pause
