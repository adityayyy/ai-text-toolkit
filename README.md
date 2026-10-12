# ai-text-toolkit

> **Note:** This project was built for personal learning — to understand how
> AI-text detection and local LLM inference work under the hood. It is not a
> polished product and should not be relied on for high-stakes decisions
> (e.g. academic integrity enforcement, content moderation, or anything where
> a wrong call has real consequences). Detection accuracy is meaningfully
> lower against current-generation AI models (see Known Limitations below) —
> treat results as a signal, not a verdict.

Fully local, free, open-source **AI text detector + AI humanizer**.
Runs on a CPU-only laptop (8-16GB RAM). No paid APIs, no cloud dependencies.

## What it does

| Part | Status | Description |
|------|--------|-------------|
| **1. Detector** | ✅ Done | GPT-2 small perplexity + burstiness + 14-dim stylometrics → LogisticRegression classifier |
| **2. Humanizer** | ✅ Done | Swappable backend (local GGUF via llama-cpp-python / OpenAI-compatible API), re-detects after rewrite |
| **3. API** | ✅ Done | FastAPI: `POST /detect`, `POST /humanize` (chains to detector) |
| **4. Frontend** | ✅ Done | Hand-drawn sketch style HTML/CSS/JS (Kalam font, dot grid, sticky-note buttons) |

---

## Quick Start (One Command)

```powershell
# Clone and enter
git clone <your-repo-url>
cd ai-text-toolkit

# One-click setup: installs deps, downloads model, trains detector, starts API
.\run_all.ps1
```

That's it. The script:
- Installs Python, git, VS Build Tools, CMake if missing
- Installs all Python dependencies
- Downloads GGUF model (~1.3 GB)
- Trains detector on HC3 (~10-20 min, skipped if `models/detector_logreg.joblib` exists)
- Starts the API on `http://localhost:8000`
- Serves the frontend on `http://localhost:3000` in a second window and opens it

> **Serve the frontend over HTTP — do not open `web/index.html` directly.**
> The page POSTs to `http://localhost:8000` (see `API_BASE_URL` in
> `web/index.html`), so opening it as a `file://` path throws `NetworkError`
> on Detect. Both `run_all` scripts handle this for you.

---

## Manual Setup

```powershell
cd ai-text-toolkit

# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Download GGUF model for local humanizer
# Automatically done by run_all.ps1, or manually:
Invoke-WebRequest "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" -OutFile "models\qwen2.5-1.5b-instruct-q4_k_m.gguf"

# 3. Train detector (one-time, ~10-20 min)
python -m detector.train

# 4. Start API
python -m api.main
# → http://localhost:8000

# 5. Serve the frontend (separate window) and open it
python -m http.server 3000 --directory web  # → open http://localhost:3000
```

---

## One-Click Scripts

| Script | Platform | Features |
|--------|----------|----------|
| `run_all.ps1` | PowerShell (recommended) | Auto-installs Python, git, VS Build Tools, CMake, downloads model, trains, starts API |
| `run_all.bat` | Command Prompt | Simpler version, installs core deps, trains, starts API |

**Options for `run_all.ps1`:**
```powershell
.\run_all.ps1                    # Full run
.\run_all.ps1 -SkipTrain         # Skip detector training
.\run_all.ps1 -SkipModel         # Skip GGUF download
.\run_all.ps1 -SkipDevTools      # Skip dev tools install
.\run_all.ps1 -Model "Phi-3-mini-4k-instruct-q4_k_m.gguf"  # Different model
```

---

## GPU Support

### Detector (PyTorch) — Auto-detects:
| Device | Package | `device=` |
|--------|---------|-----------|
| NVIDIA CUDA | built-in | `"cuda"` |
| Apple MPS | built-in | `"mps"` |
| **AMD/Intel on Windows (DirectML)** | `pip install torch-directml` | `"dml"` |
| **Intel Arc/iGPU (XPU)** | `pip install intel-extension-for-pytorch` | `"xpu"` |
| AMD ROCm (Linux) | built-in | `"cuda"` |
| CPU fallback | — | `"cpu"` |

### Humanizer (llama.cpp) — Build-time:
| Target | Build Command |
|--------|---------------|
| NVIDIA CUDA | `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python` |
| AMD ROCm (Linux) | `CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python` |
| **AMD/Intel on Windows (DirectML)** | `CMAKE_ARGS="-DLLAMA_DIRECTML=on" pip install llama-cpp-python` |
| **Pre-built DirectML wheel** | `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/directml` |
| OpenCL (any GPU/iGPU) | `CMAKE_ARGS="-DLLAMA_OPENCL=on" pip install llama-cpp-python` |

For Windows integrated GPU: use DirectML wheel:
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/directml
```

---

## Humanizer Backend

**Local GGUF (fully offline):**
```powershell
$env:HUMANIZER_BACKEND = "local"
$env:HUMANIZER_MODEL_PATH = "C:\...\models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
```

**API Backend (Ollama/LM Studio — no compilation):**
```powershell
$env:HUMANIZER_BACKEND = "api"
$env:HUMANIZER_API_URL = "http://localhost:11434/v1"   # Ollama default
$env:HUMANIZER_API_MODEL = "qwen2.5:1.5b-instruct"
```

Then restart API: `python -m api.main`

---

## API Reference

Start server:
```powershell
python -m api.main
# → http://localhost:8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/detect` | POST | Analyze text |
| `/humanize` | POST | Humanize + re-detect |

**`POST /detect`**
```json
{ "text": "Your text here..." }
```
Returns: `ai_probability`, `is_ai`, `label`, `feature_breakdown`, `sentence_scores`, `elapsed_seconds`

**`POST /humanize`**
```json
{ "text": "Your text...", "re_detect": true, "temperature": 0.7 }
```
Returns: `original_text`, `humanized_text`, `original_ai_probability`, `humanized_ai_probability`, `backend_used`, `elapsed_seconds`

---

## Detector Details

**Signals** (all local, no external API):
- **Perplexity** — GPT-2 small (~124M params) per-token log-likelihood
- **Burstiness** — Variance of per-sentence perplexity
- **14 Stylometric features**: sentence stats, TTR, punctuation, bigram uniqueness, word length, POS distribution (nltk)

**Classifier**: `StandardScaler → LogisticRegression` (14-dim vector)
- Trained on **HC3** (Hello-SimpleAI/HC3 on HuggingFace)
- Offline fallback: synthetic paragraphs (flagged as LOW accuracy)
- Model: `models/detector_logreg.joblib`
- Metrics: `models/detector_metrics.json` (acc=88%, F1=0.90, ROC-AUC=0.96)

**Performance**: ~1.4s for 500 words on CPU

---

## Project Structure

```
ai-text-toolkit/
├── detector/           # Detection engine
│   ├── __init__.py
│   ├── perplexity.py   # GPT-2 perplexity + burstiness
│   ├── stylometric.py  # Hand-computed features + AI clichés
│   ├── classifier.py   # StandardScaler + LogisticRegression
│   ├── detector.py     # Main AIDetector pipeline
│   ├── train.py        # HC3 training + CV + metrics
│   └── evaluate.py     # Accuracy report
├── humanizer/          # Text rewriting engine
│   ├── __init__.py
│   └── humanizer.py    # Swappable backends, re-detect chain
├── api/                # FastAPI backend
│   ├── __init__.py
│   └── main.py         # /detect, /humanize, /health
├── web/                # Frontend
│   └── index.html      # Hand-drawn sketch style
├── models/             # Trained model + GGUF weights
│   ├── detector_logreg.joblib
│   └── detector_metrics.json
├── run_all.ps1         # One-click full setup (PowerShell)
├── run_all.bat         # One-click full setup (CMD)
├── requirements.txt
├── CONTEXT.md
└── README.md
```

## Known Limitations

- Trained on HC3 (2022-2023 ChatGPT vs. human text) — accuracy is meaningfully
  lower against current-generation models (Claude, GPT-4/5-class, Gemini),
  which is a known industry-wide limitation, not unique to this project.
- Per-sentence AI scoring uses a heuristic (perplexity z-score + cliché bonus),
  not a dedicated classifier — see code comments for upgrade path.
- This is not a substitute for a commercial detection service and should not
  be used as the sole basis for accusations of AI use.
- GPT-2's 1024-token context window means documents longer than ~750 words
  are truncated for the perplexity/burstiness signal specifically (stylometric
  features still process the full document).

---

## Key Design Decisions

- **GPT-2 small** for perplexity: fastest CPU option (~124M params). Swap via `PerplexityComputer(model_name=...)`.
- **Single document-level classifier**: StandardScaler + LogisticRegression. Upgrade: swap inside same Pipeline.
- **14-dim fixed feature vector**: `FEATURE_NAMES` in `detector.py` is single source of truth.
- **Per-sentence AI score = heuristic** (perplexity z-score + cliché bonus). Flagged in code; upgrade path = per-sentence classifier.
- **HC3 dataset** loaded via raw JSONL (bypasses deprecated loading script in datasets 5.x).
- **Auto GPU detection**: CUDA / MPS / DirectML / Intel XPU / CPU fallback.

---

## Troubleshooting

### Windows-Specific
| Issue | Fix |
|-------|-----|
| `llama-cpp-python` build fails | Use pre-built wheel: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` or `conda install -c conda-forge llama-cpp-python` |
| `torchvision` import error | `pip uninstall torchvision` (not needed; broken with torch 2.13+) |
| `datasets` load fails | Fixed in `train.py` — uses raw JSONL via `huggingface_hub` |
| Path with spaces | Always quote: `cd "C:\path\ai detector\ai-text-toolkit"` |
| Humanizer 503 error | Set `HUMANIZER_BACKEND` + model path or API URL |

### GPU on Windows
- **Integrated/AMD GPU**: Install DirectML wheel → `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/directml`
- **Detector on AMD/Intel GPU**: `pip install torch-directml` then use `device="dml"`

### General
| Issue | Fix |
|-------|-----|
| HC3 load fails | Check internet; offline fallback activates with warning |
| Detector not trained | Run `python -m detector.train` first |
| Humanizer 503 | Set `HUMANIZER_BACKEND` + model path or API URL |
| Port 8000 in use | `uvicorn api.main:app --port 8001` |

---

## License

MIT — free, open-source, no warranty.

---

Built for learning. Comments throughout explain the "why". Swap points marked for easy upgrades.