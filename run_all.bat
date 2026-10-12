@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

title ai-text-toolkit Setup
echo ============================================================
echo  ai-text-toolkit - Full Setup and Run
echo ============================================================
echo.

REM ============================================================
REM 1. Install core requirements
REM ============================================================
echo ============================================================
echo  [1/7] Installing core requirements...
echo  (torch is a large download, 600MB-2GB - this can take several
echo   minutes with little visible output. This is normal, please wait.)
echo ============================================================
pip install "torch>=2.0.0" "transformers>=4.35.0" "nltk>=3.8.0" "scikit-learn>=1.3.0" "fastapi>=0.109.0" "uvicorn[standard]>=0.27.0" "pydantic>=2.7.0" "pydantic-settings>=2.1.0" "datasets>=2.15.0" "joblib>=1.3.0" "httpx" "typing-extensions>=4.8.0"
if errorlevel 1 (
    echo WARNING: Some core packages may have failed. Check output above.
)

REM ============================================================
REM 2. Fix huggingface-hub version (transformers needs <2.0)
REM ============================================================
echo.
echo ============================================================
echo  [2/7] Fixing huggingface-hub version (transformers compat)...
echo ============================================================
pip install "huggingface-hub>=1.5.0,<2.0" --force-reinstall 2>NUL | findstr /V "WARNING"
if errorlevel 1 (
    echo WARNING: huggingface-hub fix may have issues.
)

REM ============================================================
REM 3. Install llama-cpp-python (pre-built CPU wheel)
REM ============================================================
echo.
echo ============================================================
echo  [3/7] Installing llama-cpp-python (humanizer backend)...
echo ============================================================
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
if errorlevel 1 (
    echo WARNING: llama-cpp-python install failed. Humanizer local backend
    echo will not work until this is resolved manually. You can retry with:
    echo   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
)

REM ============================================================
REM 4. Download NLTK POS tagger data
REM ============================================================
echo.
echo ============================================================
echo  [4/7] Downloading NLTK data...
echo ============================================================
python scripts\download_nltk_data.py
if errorlevel 1 (
    echo WARNING: NLTK data download failed. POS features may not work.
)

REM ============================================================
REM 5. Download humanizer GGUF model
REM ============================================================
echo.
echo ============================================================
echo  [5/7] Checking humanizer model...
echo ============================================================
if not exist models mkdir models

set EXISTING_MODEL=
for %%F in (models\*.gguf) do (
    if not defined EXISTING_MODEL set EXISTING_MODEL=%%F
)

if defined EXISTING_MODEL (
    echo [OK] Found existing model: !EXISTING_MODEL!
    echo Skipping download.
    set HUMANIZER_MODEL_PATH=%~dp0!EXISTING_MODEL!
) else (
    echo No .gguf model found in models\ - downloading Qwen2.5-1.5B-Instruct Q4_K_M (~1GB)...
    python -c "from huggingface_hub import hf_hub_download; import shutil; p = hf_hub_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF', filename='qwen2.5-1.5b-instruct-q4_k_m.gguf'); shutil.copy(p, r'models\qwen2.5-1.5b-instruct-q4_k_m.gguf')"
    if errorlevel 1 (
        echo WARNING: Model download failed. Humanizer will not work until
        echo this is resolved. Re-run this script to retry, or download
        echo manually from huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF
    ) else (
        echo [OK] Model downloaded.
        set HUMANIZER_MODEL_PATH=%~dp0models\qwen2.5-1.5b-instruct-q4_k_m.gguf
    )
)
set HUMANIZER_BACKEND=local

REM ============================================================
REM 6. Train detector (using new module structure)
REM ============================================================
echo.
echo ============================================================
echo  [6/7] Training detector...
echo ============================================================
if exist models\detector_logreg.joblib (
    echo [OK] Detector already trained. Skipping.
) else (
    echo Training on HC3 dataset (500 train + 150 test per class)...
    echo This takes ~10-20 minutes on CPU. Please wait...
    python -m core.detector.training
    if errorlevel 1 (
        echo ERROR: Training failed.
        pause
        exit /b 1
    )
    echo.
    echo Training complete! Model saved to models\detector_logreg.joblib
)

REM ============================================================
REM 7. Start API server + open frontend in browser
REM ============================================================
echo.
echo ============================================================
echo  [7/7] Starting API server and opening frontend...
echo ============================================================
echo.
echo API:      http://localhost:8000  (docs at /docs)
echo Frontend: http://localhost:3000
echo.
echo Press CTRL+C in this window to stop the server.
echo ============================================================
echo.

REM cwd is already %~dp0 (line 3), so a relative path avoids space-quoting bugs
start "AI Toolkit - Frontend :3000" cmd /k python -m http.server 3000 --directory web
start "" cmd /c "timeout /t 3 /nobreak >nul && start "" http://localhost:3000"
python -m api.main

echo.
echo Server stopped. Press any key to exit...
pause