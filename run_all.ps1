<#>
.SYNOPSIS
    ai-text-toolkit — One-click setup & run (PowerShell)
    Installs deps, trains detector, starts API server
    Auto-installs Python, git, VS Build Tools if missing (non-dev machines)

.DESCRIPTION
    Runs the complete pipeline:
    1. Installs Python, git, VS Build Tools if missing (non-dev machines)
    2. Installs Python dependencies (core + llama-cpp-python)
    3. Downloads NLTK data
    4. Fixes huggingface-hub version compatibility
    5. Trains detector on HC3 (if not already trained)
    6. Starts FastAPI server on http://localhost:8000

.EXAMPLE
    .\run_all.ps1

.NOTES
    Run from project root: C:\Users\agane\OneDrive\Desktop\ai detector\ai-text-toolkit
    Run as Administrator for best results (needed for VS Build Tools, winget)
#>

param(
    [switch]$SkipTrain,
    [switch]$SkipLlamaCpp,
    [switch]$SkipDevTools,
    [switch]$ForceReinstall,
    [string]$Model = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    [switch]$SkipModel
)

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $PROJECT_ROOT

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ai-text-toolkit — Full Setup & Run (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ─────────────────────────────────────────────────────────────────
# 0. DEVELOPMENT TOOLS — Install if missing (for non-dev machines)
# ─────────────────────────────────────────────────────────────────
if (-not $SkipDevTools) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  [0/7] Checking development tools..." -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Cyan

    # --- Python ---
    function Test-Python {
        try {
            $ver = python --version 2>&1
            if ($ver -match "Python 3\.(1[0-9]|[2-9]\d)") { return $true }
        } catch {}
        return $false
    }

    if (-not (Test-Python)) {
        Write-Host "  Python not found or too old. Installing via winget..." -ForegroundColor Yellow
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id Python.Python.3.12 --silent --accept-source-agreements --accept-package-agreements
            $env:PATH = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
            refreshenv 2>$null
        } else {
            Write-Warning "winget not available. Please install Python 3.12+ from https://python.org (check 'Add to PATH')"
            Write-Warning "Then re-run this script."
            exit 1
        }
    } else {
        Write-Host "  [OK] Python: $(python --version 2>&1)" -ForegroundColor Green
    }

    # --- git ---
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "  git not found. Installing via winget..." -ForegroundColor Yellow
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id Git.Git --silent --accept-source-agreements --accept-package-agreements
        } else {
            Write-Warning "git not found and winget unavailable. Install from https://git-scm.com"
        }
    } else {
        Write-Host "  [OK] git: $(git --version)" -ForegroundColor Green
    }

    # --- Visual Studio Build Tools (for llama-cpp-python compilation if needed) ---
    function Test-VSBuildTools {
        $vsPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        if (Test-Path $vsPath) { return $true }
        $vsPath2 = "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
        if (Test-Path $vsPath2) { return $true }
        return $false
    }

    if (-not (Test-VSBuildTools)) {
        Write-Host "  Visual Studio Build Tools not found. Installing (required for llama-cpp-python if pip wheel fails)..." -ForegroundColor Yellow
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Host "  Installing VS Build Tools with C++ workload... (this may take 10-20 min)" -ForegroundColor Gray
            winget install --id Microsoft.VisualStudio.2022.BuildTools --silent --accept-source-agreements --accept-package-agreements --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        } else {
            Write-Warning "VS Build Tools not found and winget unavailable."
            Write-Warning "  Install from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022"
            Write-Warning "  Select 'Desktop development with C++' workload."
        }
    } else {
        Write-Host "  [OK] VS Build Tools found" -ForegroundColor Green
    }

    # --- CMake (for llama-cpp-python) ---
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        Write-Host "  CMake not found. Installing via winget..." -ForegroundColor Yellow
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id Kitware.CMake --silent --accept-source-agreements --accept-package-agreements
        } else {
            Write-Warning "CMake not found. Install from https://cmake.org/download/"
        }
    } else {
        Write-Host "  [OK] CMake: $(cmake --version | Select-Object -First 1)" -ForegroundColor Green
    }

    Write-Host "  Development tools check complete." -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ai-text-toolkit — Full Setup & Run (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ─────────────────────────────────────────────────────────────────
# 1. PYTHON DEPENDENCIES
# ─────────────────────────────────────────────────────────────────
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  [1/7] Installing core Python requirements..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

$corePkgs = @(
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "nltk>=3.8.0",
    "scikit-learn>=1.3.0",
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.7.0",
    "datasets>=2.15.0",
    "joblib>=1.3.0",
    "httpx"
)
if ($ForceReinstall) { $corePkgs = $corePkgs | ForEach-Object { "--force-reinstall $_" } }
$installArgs = @("install") + $corePkgs
& pip @installArgs
if ($LASTEXITCODE -ne 0) { Write-Warning "Some core packages failed. Continuing..." }

# ─────────────────────────────────────────────────────────────────
# 2. llama-cpp-python (pre-built CPU wheel - no compiler needed)
# ─────────────────────────────────────────────────────────────────
if (-not $SkipLlamaCpp) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  [2/7] Installing llama-cpp-python (pre-built CPU wheel)..." -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Cyan

    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "llama-cpp-python install failed. Trying conda..."
        if (Get-Command conda -ErrorAction SilentlyContinue) {
            conda install -c conda-forge -y llama-cpp-python
        } else {
            Write-Warning "llama-cpp-python install failed. Humanizer local backend may not work."
            Write-Warning "  Fix later: conda install -c conda-forge llama-cpp-python"
        }
    }
}

# ─────────────────────────────────────────────────────────────────
# 3. NLTK POS tagger data
# ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  [3/7] Downloading NLTK POS tagger data..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

python "$PROJECT_ROOT\scripts\download_nltk_data.py"

# ─────────────────────────────────────────────────────────────────
# 4. Fix huggingface-hub version (transformers needs <2.0)
# ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  [4/8] Fixing huggingface-hub version (transformers compat)..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
pip install "huggingface-hub>=1.5.0,<2.0" --force-reinstall 2>&1 | Where-Object { $_ -notlike "*WARNING*" }

# ─────────────────────────────────────────────────────────────────
# 5. Download Hugging Face GGUF model (for local humanizer)
# ─────────────────────────────────────────────────────────────────
if (-not $SkipModel) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  [5/8] Downloading GGUF model: $Model..." -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Cyan

    $modelPath = Join-Path $PROJECT_ROOT "models\$Model"
    if (Test-Path $modelPath) {
        Write-Host "  [OK] Model already exists: $Model" -ForegroundColor Green
    } else {
        Write-Host "  Downloading $Model (~1-2 GB)..." -ForegroundColor Gray
        $urls = @(
            "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/$Model",
            "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/$Model"
        )
        $downloaded = $false
        foreach ($url in $urls) {
            try {
                Write-Host "  Trying: $url" -ForegroundColor Gray
                Invoke-WebRequest -Uri $url -OutFile $modelPath -ErrorAction Stop
                $downloaded = $true
                break
            } catch {
                Write-Host "  Failed: $url" -ForegroundColor Yellow
            }
        }
        if (-not $downloaded) {
            Write-Warning "Failed to download $Model from known repos."
            Write-Warning "  You can download manually from HuggingFace and place in models\"
            Write-Warning "  Example: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF"
        } else {
            Write-Host "  [OK] Model saved to: $modelPath" -ForegroundColor Green
        }
    }
} else {
    Write-Host "  Skipping model download (--SkipModel specified)" -ForegroundColor Yellow
}

# ─────────────────────────────────────────────────────────────────
# 6. Train detector
# ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  [5/7] Checking detector model..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

$modelPath = Join-Path $PROJECT_ROOT "models\detector_logreg.joblib"
if (Test-Path $modelPath -and -not $SkipTrain) {
    Write-Host "[OK] Detector already trained. Skipping..." -ForegroundColor Green
} elseif ($SkipTrain) {
    Write-Host "Skipping training (--SkipTrain specified)" -ForegroundColor Yellow
} else {
    Write-Host "Training detector on HC3 (500 train + 150 test per class)..." -ForegroundColor Yellow
    Write-Host "This takes ~10-20 minutes on CPU. Please wait..." -ForegroundColor Gray
    python -m detector.train
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Training failed."
        exit 1
    }
    Write-Host ""
    Write-Host "Training complete! Model saved to models\detector_logreg.joblib" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────────────────
# 7. Fix huggingface-hub version again (in case training upgraded it)
# ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  [7/8] Re-verifying huggingface-hub version..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
pip install "huggingface-hub>=1.5.0,<2.0" --force-reinstall 2>&1 | Where-Object { $_ -notlike "*WARNING*" }

# ─────────────────────────────────────────────────────────────────
# 8. Start API server + serve frontend
# ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  [7/7] Starting API server on http://localhost:8000" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Point the humanizer at whichever GGUF model is in models\ (else /humanize 503s)
$env:HUMANIZER_BACKEND = "local"
$gguf = Get-ChildItem (Join-Path $PROJECT_ROOT "models\*.gguf") -ErrorAction SilentlyContinue | Select-Object -First 1
if ($gguf) {
    $env:HUMANIZER_MODEL_PATH = $gguf.FullName
    Write-Host "Humanizer model: $($gguf.Name)" -ForegroundColor Green
} else {
    Write-Warning "No .gguf model in models\ - /humanize will return 503."
}

Write-Host "API endpoints:" -ForegroundColor Green
Write-Host "  POST http://localhost:8000/detect" -ForegroundColor White
Write-Host "  POST http://localhost:8000/humanize" -ForegroundColor White
Write-Host "  GET  http://localhost:8000/health" -ForegroundColor White
Write-Host ""

# Serve the frontend in its own window, then open it. The page MUST be served
# over HTTP - opening web\index.html as file:// throws NetworkError on Detect,
# because it POSTs to http://localhost:8000.
Start-Process cmd -ArgumentList '/k', 'python -m http.server 3000 --directory web' -WorkingDirectory $PROJECT_ROOT
Start-Sleep -Seconds 3
Start-Process "http://localhost:3000"

Write-Host "Press CTRL+C to stop the API server (close the other window to stop the frontend)." -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

python -m api.main