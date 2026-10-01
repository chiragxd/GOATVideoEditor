@echo off
setlocal EnableDelayedExpansion
title GOAT Studio - Installer
cd /d "%~dp0"
echo.
echo  =====================================================
echo    GOAT STUDIO  -  one-click installer (free + local)
echo  =====================================================
echo.

rem ---- 1. Python (3.11 preferred: every AI library supports it) -------------------
set "PY="
for %%V in (3.11 3.12 3.13 3.14) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    echo [1/6] Installing Python 3.11 ...
    winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements --scope user
    set "PY=py -3.11"
) else (
    echo [1/6] Using Python: !PY!
)

rem ---- 2. Virtual environment ------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [2/6] Creating virtual environment ...
    !PY! -m venv .venv || goto :fail
) else (
    echo [2/6] Virtual environment exists.
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip -q

rem ---- 3. PyTorch: CUDA build if an NVIDIA GPU is present, else CPU ---------------
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo [3/6] NVIDIA GPU found - installing CUDA PyTorch ...
    "%VPY%" -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121 || goto :fail
) else (
    echo [3/6] No NVIDIA GPU - installing CPU PyTorch ...
    "%VPY%" -m pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu || goto :fail
)

rem ---- 4. App libraries ------------------------------------------------------------
echo [4/6] Installing app libraries ...
"%VPY%" -m pip install -q -r requirements.txt || goto :fail
rem Studio voice cleanup (DeepFilterNet) lives in its own Python 3.11 env: it needs older torch/numpy.
if not exist "tools\dfenv\Scripts\python.exe" (
    py -3.11 -c "import sys" >nul 2>&1 || winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements --scope user
    py -3.11 -m venv tools\dfenv && (
        echo        Setting up studio voice cleanup ^(DeepFilterNet^) ...
        tools\dfenv\Scripts\python -m pip install -q --upgrade pip
        tools\dfenv\Scripts\python -m pip install -q torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
        tools\dfenv\Scripts\python -m pip install -q deepfilternet soundfile "numpy<2"
    ) || echo        Voice cleanup sidecar skipped - noisereduce will be used.
)

rem ---- 5. Fonts and small models ---------------------------------------------------
echo [5/6] Downloading fonts and the person-cutout model ...
if not exist fonts mkdir fonts
if not exist models mkdir models
for %%F in ("ofl/montserrat/Montserrat%%5Bwght%%5D.ttf|Montserrat[wght].ttf" "ofl/poppins/Poppins-Black.ttf|Poppins-Black.ttf" "ofl/poppins/Poppins-ExtraBold.ttf|Poppins-ExtraBold.ttf" "ofl/anton/Anton-Regular.ttf|Anton-Regular.ttf" "ofl/bebasneue/BebasNeue-Regular.ttf|BebasNeue-Regular.ttf") do (
    for /f "tokens=1,2 delims=|" %%A in (%%F) do (
        if not exist "fonts\%%B" curl -sfL -o "fonts\%%B" "https://github.com/google/fonts/raw/main/%%A"
    )
)
if not exist "models\u2netp.onnx" curl -sfL -o "models\u2netp.onnx" "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"

rem ---- 6. Local AI (Ollama + models) -----------------------------------------------
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not exist "%OLLAMA%" (
    echo [6/6] Installing Ollama ...
    winget install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements
)
if exist "%OLLAMA%" (
    echo [6/6] Downloading AI models - qwen2.5:7b ~4.7 GB and moondream ~1.7 GB, first time only ...
    "%OLLAMA%" pull qwen2.5:7b
    "%OLLAMA%" pull moondream
) else (
    echo [6/6] Ollama not found - the app will use its rule-based director until you install it from ollama.com
)

echo.
echo  Done!  Start the app with  "Start GOAT Studio.bat"
echo.
pause
exit /b 0

:fail
echo.
echo  Installation failed - see the messages above.
pause
exit /b 1
