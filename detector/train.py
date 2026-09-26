"""
Training script for the AI-text detector.

Usage (from the project root):
    python -m detector.train                # HC3 data, 500 train + 150 test per class
    python -m detector.train --n 100        # smaller/faster run
    python -m detector.train --test 50      # smaller held-out test set

WHAT IT DOES:
1. Loads the HC3 dataset (Human ChatGPT Comparison Corpus, Hello-SimpleAI/HC3
   on HuggingFace) - real human answers vs real ChatGPT answers.
   OFFLINE FALLBACK: if HF/datasets is unreachable, uses bundled synthetic
   samples (LOW accuracy - tiny and hand-labelled; retrain with HC3 when
   online; the fallback is announced loudly when used).
2. Extracts features for every sample (the slow part: one GPT-2 forward pass
   per document, ~0.5-1s each on CPU  ->  ~10-20 min for the default 1300
   documents, one-time only).
3. Trains StandardScaler + LogisticRegression, saves to models/detector_logreg.joblib.
4. Evaluates on the held-out test set (accuracy/precision/recall/F1/ROC-AUC/
   confusion matrix) AND runs 5-fold cross-validation on the training features
   (nearly free - LogReg trains in milliseconds). Saves everything to
   models/detector_metrics.json.
"""

import argparse
import json
import os
import random
import time

import numpy as np

from detector.perplexity import PerplexityComputer, compute_burstiness
from detector.stylometric import compute_stylometric_features
from detector.classifier import AIDetectorClassifier, DETECTOR_PKL, MODELS_DIR
from detector.detector import FEATURE_NAMES

METRICS_JSON = os.path.join(MODELS_DIR, "detector_metrics.json")

# Ignore answers shorter than this many words - very short samples are noisy
# for stylometric features and would hurt the classifier.
MIN_WORDS = 40


# ── Data loading ─────────────────────────────────────────────────────
def load_hc3(n_train: int, n_test: int) -> dict:
    """Load HC3 from HuggingFace raw JSONL. Returns {"human_train", "ai_train",
    "human_test", "ai_test"} - lists of document strings.

    Uses raw JSONL (all.jsonl) because datasets 5.x dropped support for
    loading scripts (HC3.py). This works offline once cached.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    print("[data] downloading/loading HC3 (Hello-SimpleAI/HC3) raw JSONL...")
    # Download the raw all.jsonl (or all.jsonl.gz if it exists)
    path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    ds = load_dataset("json", data_files=path, split="train")

    humans, ais = [], []
    for row in ds:
        humans.extend(row["human_answers"])
        ais.extend(row["chatgpt_answers"])

    humans = [h for h in humans if len(h.split()) >= MIN_WORDS]
    ais = [a for a in ais if len(a.split()) >= MIN_WORDS]
    print(f"[data] usable samples: {len(humans)} human, {len(ais)} ai")

    rng = random.Random(42)          # fixed seed -> reproducible split
    rng.shuffle(humans)
    rng.shuffle(ais)

    need = n_train + n_test
    return {
        "human_train": humans[:n_train],
        "human_test": humans[n_train:need],
        "ai_train": ais[:n_train],
        "ai_test": ais[n_train:need],
    }


def load_synthetic(n_train: int, n_test: int) -> dict:
    """OFFLINE FALLBACK - hand-written human-style vs AI-style paragraphs,
    replicated to fill the requested counts.

    ponytail: LOW ACCURACY by construction (tiny, hand-labelled, no real
    human variance). Upgrade path: rerun `python -m detector.train` with
    internet access so it picks up HC3.
    """
    human_style = [
        ("Honestly I don't know why this keeps happening. I set three alarms, "
         "one across the room, and I still slept through all of them. My "
         "roommate said she heard the third one ring for a full minute. "
         "Anyway I made it to the lecture twenty minutes late and the prof "
         "just looked at me. Not my best morning."),
        ("Been messing with a sourdough starter for like two months now. First "
         "attempt smelled like nail polish so I tossed it. Second one actually "
         "worked and the bread came out okay, a bit flat but tasty. My mom "
         "thinks I should just buy bread. She's probably right but here we are."),
        ("So the job interview was yesterday. I thought it went fine until they "
         "asked about my thesis and I blanked on my own research for a second. "
         "Recovered-ish. They said they'd get back to me next week which "
         "usually means no, but the guy shook my hand with both hands so who "
         "knows. Fingers crossed I guess."),
        ("We hiked up at 4am to catch the sunrise which sounded romantic until "
         "we were halfway up in the dark and my flashlight died. My friend's "
         "phone had 8% battery. We made it, barely, and the view was unreal. "
         "Would I do it again? Ask me next week when my legs stop hurting."),
    ]
    ai_style = [
        ("It is important to note that establishing a consistent morning "
         "routine is essential for productivity and overall well-being. "
         "Furthermore, utilizing multiple alarm systems can significantly "
         "reduce the likelihood of oversleeping. Additionally, effective time "
         "management strategies enable individuals to navigate daily "
         "challenges with confidence. In conclusion, small structural changes "
         "can yield substantial improvements in daily outcomes."),
        ("Embarking on the journey of artisanal bread-making offers numerous "
         "rewards for dedicated home bakers. Moreover, maintaining a healthy "
         "sourdough starter requires careful attention to feeding schedules "
         "and environmental conditions. It can be argued that homemade bread "
         "provides superior nutritional value compared to commercial "
         "alternatives. In essence, patience and consistency are the key "
         "ingredients for success in this rewarding endeavor."),
        ("Preparing thoroughly for a job interview is a crucial step toward "
         "career advancement. Furthermore, articulating one's research and "
         "professional experience clearly demonstrates subject matter "
         "expertise. Additionally, non-verbal communication plays a "
         "significant role in forming positive impressions. In conclusion, a "
         "combination of preparation and confident presentation maximizes the "
         "likelihood of a favorable outcome."),
        ("Experiencing a sunrise from a mountain summit offers profound "
         "rewards for those willing to rise early. Nevertheless, adequate "
         "preparation, including reliable lighting equipment, is essential "
         "for safety during pre-dawn hikes. Additionally, shared challenges "
         "often strengthen interpersonal bonds between hiking partners. In "
         "summary, the physical demands of such endeavors are outweighed by "
         "the memorable experiences they provide."),
    ]

    def fill(samples: list[str], n: int) -> list[str]:
        out = []
        while len(out) < n:
            out.extend(samples)
        return out[:n]

    return {
        "human_train": fill(human_style, n_train),
        "human_test": fill(human_style, n_test),
        "ai_train": fill(ai_style, n_train),
        "ai_test": fill(ai_style, n_test),
    }


# ── Feature extraction ───────────────────────────────────────────────
def extract_all(lm: PerplexityComputer, texts: list[str], label: int) -> tuple[list, list]:
    """Extract the feature vector for every text. Returns (X, y)."""
    X, y = [], []
    t0 = time.time()
    for i, t in enumerate(texts, 1):
        res = lm.analyze(t)
        burst = compute_burstiness(res["sentence_ppls"])
        styl = compute_stylometric_features(t)
        values = {"perplexity": res["perplexity"], "burstiness": burst, **styl}
        X.append([values[n] for n in FEATURE_NAMES])
        y.append(label)
        if i % 50 == 0 or i == len(texts):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  [{label and 'ai' or 'human'}] {i}/{len(texts)} docs  ({rate:.1f} docs/s)")
    return X, y


# ── Evaluation ───────────────────────────────────────────────────────
def evaluate(clf: AIDetectorClassifier, X_test, y_test) -> dict:
    """Held-out test metrics. Prints a classification report, returns metrics."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix,
                                 classification_report)

    proba = clf.predict_proba(np.asarray(X_test))
    pred = (proba >= 0.5).astype(int)
    y_test = np.asarray(y_test)

    metrics = {
        "n_test": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    if len(set(y_test)) == 2:                      # ROC-AUC needs both classes
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)

    print("\n=== Held-out test set ===")
    print(classification_report(y_test, pred, target_names=["human", "ai"],
                                zero_division=0))
    return metrics


def evaluate_saved(n_test: int = 150) -> dict:
    """Evaluate the SAVED classifier on the HC3 held-out test sample.
    This is how you answer 'is the detector accurate?' after training."""
    clf = AIDetectorClassifier()
    clf.load()

    try:
        data = load_hc3(0, n_test)
        source = "HC3"
    except Exception as e:
        print(f"[data] HC3 unavailable ({e.__class__.__name__}: {e}); using synthetic fallback")
        data = load_synthetic(0, n_test)
        source = "synthetic (LOW accuracy)"

    lm = PerplexityComputer()
    print(f"[evaluate] extracting features for test set ({source})...")
    Xh, _ = extract_all(lm, data["human_test"], 0)
    Xa, _ = extract_all(lm, data["ai_test"], 1)
    y = [0] * len(data["human_test"]) + [1] * len(data["ai_test"])
    return evaluate(clf, Xh + Xa, y)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train the AI-text detector")
    ap.add_argument("--n", type=int, default=500, help="train samples per class")
    ap.add_argument("--test", type=int, default=150, help="test samples per class")
    args = ap.parse_args()

    try:
        data = load_hc3(args.n, args.test)
        source = "HC3 (HuggingFace)"
    except Exception as e:
        print(f"[data] HC3 unavailable ({e.__class__.__name__}: {e})")
        print("[data] FALLING BACK TO SYNTHETIC DATA - accuracy will be LOW.")
        print("[data] Retrain with internet access to pick up HC3.")
        data = load_synthetic(args.n, args.test)
        source = "synthetic fallback (LOW accuracy)"

    lm = PerplexityComputer()   # downloads GPT-2 on first run (~500MB, one-time)
    print(f"\n[train] extracting features ({source}) - the slow part, one-time...")
    t0 = time.time()
    Xh, yh = extract_all(lm, data["human_train"], 0)
    Xa, ya = extract_all(lm, data["ai_train"], 1)
    X = np.array(Xh + Xa, dtype=np.float32)
    y = np.array(yh + ya)
    print(f"[train] feature extraction took {time.time() - t0:.0f}s for {len(y)} docs")

    clf = AIDetectorClassifier()
    clf.fit(X, y)
    clf.save()

    # Held-out test metrics (features not seen in training)
    Xh_t, _ = extract_all(lm, data["human_test"], 0)
    Xa_t, _ = extract_all(lm, data["ai_test"], 1)
    metrics = evaluate(clf, Xh_t + Xa_t,
                       [0] * len(data["human_test"]) + [1] * len(data["ai_test"]))

    # 5-fold CV on training features - nearly free (LogReg trains in ms)
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(clf.pipeline, X, y, cv=5)
    metrics["cv_accuracy_mean"] = round(float(cv.mean()), 4)
    metrics["cv_accuracy_std"] = round(float(cv.std()), 4)
    metrics["n_train"] = int(len(y))
    metrics["data_source"] = source
    metrics["feature_names"] = FEATURE_NAMES

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[metrics] saved -> {METRICS_JSON}")
    print(f"[metrics] summary: test accuracy={metrics['accuracy']}, "
          f"F1={metrics['f1']}, ROC-AUC={metrics.get('roc_auc')}, "
          f"CV acc={metrics['cv_accuracy_mean']} ± {metrics['cv_accuracy_std']}")


if __name__ == "__main__":
    main()
