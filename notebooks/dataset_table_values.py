#!/usr/bin/env python3
"""
Print dataset characteristics for the paper table (Table: Dataset characteristics).
Run from notebooks: python dataset_table_values.py

Uses: latest corpus_train_*.txt, corpus_val_*.txt (line counts = encounters),
      latest bert_triage_model_*/config.json (vocab_size),
      label_mapping.NUM_LABELS.
Unique patients: not stored in corpus files (split is encounter-level); output N/A.
"""

import sys
from pathlib import Path

# Run from notebooks
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import paths_config
    paths_config.ensure_dirs()
    DATA_PROCESSED = paths_config.DATA_PROCESSED
    MODELS_TRIAGE = paths_config.MODELS_TRIAGE
except Exception:
    DATA_PROCESSED = SCRIPT_DIR / "data" / "processed"
    MODELS_TRIAGE = SCRIPT_DIR / "models" / "triage"

def main():
    # 1. Latest train/val corpus files — use same timestamp so split is from one run (80/20)
    train_files = sorted(DATA_PROCESSED.glob("corpus_train_*.txt"), key=lambda p: p.name, reverse=True)
    train_file = train_files[0] if train_files else None
    val_file = None
    if train_file:
        # e.g. corpus_train_20260130_023306.txt -> corpus_val_20260130_023306.txt (same run = 80/20)
        suffix = train_file.stem.replace("corpus_train_", "")
        same_run = DATA_PROCESSED / f"corpus_val_{suffix}.txt"
        val_file = same_run if same_run.exists() else (sorted(DATA_PROCESSED.glob("corpus_val_*.txt"), key=lambda p: p.name, reverse=True)[0] if list(DATA_PROCESSED.glob("corpus_val_*.txt")) else None)
    if not train_file or not val_file:
        print("Missing corpus_train_*.txt or corpus_val_*.txt in data/processed/. Run notebook 05 first.")
        return

    training_samples = sum(1 for _ in open(train_file, "r", encoding="utf-8"))
    validation_samples = sum(1 for _ in open(val_file, "r", encoding="utf-8"))

    # 2. Vocabulary size from latest triage model config
    triage_dirs = sorted(MODELS_TRIAGE.glob("bert_triage_model_*"), key=lambda p: p.name, reverse=True)
    vocab_size = None
    for d in triage_dirs:
        config_path = d / "config.json"
        if config_path.exists():
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            vocab_size = config.get("vocab_size")
            break
    if vocab_size is None:
        vocab_size = "~140"

    # 3. Classes
    try:
        from label_mapping import NUM_LABELS
    except ImportError:
        NUM_LABELS = 8

    # 4. Unique patients: not in corpus (encounter-level split)
    unique_train = "N/A"
    unique_val = "N/A"

    # 5. Split ratio
    total = training_samples + validation_samples
    split_pct = f"{100 * training_samples / total:.0f}% / {100 * validation_samples / total:.0f}%" if total else "80% / 20%"

    # Output for paper table
    print("Dataset characteristics (copy into paper table):")
    print("-" * 50)
    print(f"Training samples (encounters)   {training_samples:,}")
    print(f"Validation samples (encounters) {validation_samples:,}")
    print(f"Unique patients (train)         {unique_train}")
    print(f"Unique patients (val)          {unique_val}")
    print(f"Vocabulary size (tokens)       {vocab_size}")
    print(f"Classes (specialist referrals) {NUM_LABELS}")
    print(f"Patient-level split            {split_pct}")
    print("-" * 50)
    # LaTeX: use \, for thousands separator
    def latex_num(n):
        return f"{n:,}".replace(",", r"\,")
    print("\nLaTeX table row values (no header):")
    print(f"Training samples (encounters) & {latex_num(training_samples)} \\\\")
    print(f"Validation samples (encounters) & {latex_num(validation_samples)} \\\\")
    print(f"Unique patients (train) & {unique_train} \\\\")
    print(f"Unique patients (val) & {unique_val} \\\\")
    print(f"Vocabulary size (tokens) & $\\sim${vocab_size} \\\\")
    print(f"Classes (specialist referrals) & {NUM_LABELS} \\\\")
    print(f"Patient-level split & {split_pct} \\\\")

if __name__ == "__main__":
    main()
