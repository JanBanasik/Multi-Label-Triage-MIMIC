"""
Save 4 figures for the paper: attention (with token labels) + output for 2 cases.
Cases: (1) complex multi-label (5 referrals), (2) nephrology-only.
Run from notebooks after notebook 06 cells 0, 1, 2, or: cd notebooks && python save_attention_three_figs.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import paths_config
    paths_config.ensure_dirs()
    OUT_DIR = Path(paths_config.OUTPUTS_PLOTS)
except Exception:
    OUT_DIR = Path("outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_attention_with_labels(tokens, attn_matrix, out_path):
    """Attention map WITH token labels (like notebook plot_token_scores)."""
    special = {"[CLS]", "[SEP]", "[PAD]"}
    valid_indices = [i for i, t in enumerate(tokens) if t not in special]
    valid_tokens = [tokens[i] for i in valid_indices]
    scores = attn_matrix[0, valid_indices].astype(float)
    if len(scores) == 0:
        print("Skip (no valid tokens):", out_path)
        return
    if scores.max() > 0:
        scores = scores / scores.max()
    scores = scores.reshape(-1, 1)
    n = len(valid_tokens)
    fig, ax = plt.subplots(figsize=(2, max(2, 0.3 * n)))
    sns.heatmap(scores, yticklabels=valid_tokens, xticklabels=[], cmap="Reds", annot=True, fmt=".2f", cbar=False, ax=ax)
    ax.set_ylabel("Input token", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", out_path)

def save_output_figure(all_probs, class_names, expected_labels, threshold=0.15, out_path=None):
    if out_path is None:
        return
    expected_set = {s.strip().upper() for s in expected_labels.split(",")}
    def norm(c):
        return c.upper().replace("TARGET_SPEC_", "").replace("TARGET_POZ", "POZ")
    gt = [norm(c) in expected_set for c in class_names]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    y_pos = np.arange(len(class_names))[::-1]
    colors = ["#2d6a4f" if gt[i] else "#495057" for i in range(len(class_names))]
    ax.barh(y_pos, all_probs, height=0.7, color=colors, edgecolor="none")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Probability", fontsize=8)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", out_path)

# Two cases: (0) complex 5-label, (1) nephrology-only
CASES = [
    {
        "name": "complex",
        "text": "AGE_70 SEX_F HGB_Q1_CRITICAL HCT_Q1_CRITICAL PLT_Q3 MCV_Q5 WBC_Q9 CREATININE_Q9 UREA_Q9 ALT_Q4 AST_VAL COMPOSITE_ANEMIA_INDICATOR COMPOSITE_THROMBOCYTOPENIA_INDICATOR COMPOSITE_RENAL_FAILURE_INDICATOR COMPOSITE_BLEEDING_RISK COMPOSITE_INFECTION_INDICATOR COMPOSITE_SEVERE_ANEMIA SYMPTOM_GI_BLEED_NO HIST_BLOOD_THINNERS_NO SYMPTOM_WEIGHT_LOSS_NO SYMPTOM_CHEST_PAIN_NO SYMPTOM_DYSPNEA_NO",
        "expected": "HEMATOLOG, NEFROLOG, SOR, KARDIOLOG, PULMONOLOG",
    },
    {
        "name": "nephro",
        "text": "AGE_60 SEX_M HGB_Q5 HCT_Q5 PLT_Q6 MCV_Q5 WBC_Q5 CREATININE_Q9 ALT_Q5",
        "expected": "NEFROLOG",
    },
]


def main():
    try:
        import sys
        nb_dir = Path(__file__).resolve().parent
        if str(nb_dir) not in sys.path:
            sys.path.insert(0, str(nb_dir))
        import torch
        from transformers import BertForSequenceClassification, PreTrainedTokenizerFast
        import glob
        import paths_config
        paths_config.ensure_dirs()
        triage_models = sorted(glob.glob(str(paths_config.MODELS_TRIAGE / "bert_triage_model_*")), reverse=True)
        model_path = None
        for d in triage_models:
            if (Path(d) / "tokenizer.json").exists():
                model_path = d
                break
        if not model_path:
            print("Run notebook 06 first (cells 0, 1, 2). No bert_triage_model_* found.")
            return
        from label_mapping import TARGET_NAMES, NUM_LABELS
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(
            model_path, output_attentions=True, local_files_only=True,
            num_labels=NUM_LABELS, problem_type="multi_label_classification", ignore_mismatched_sizes=True
        )
        model.eval()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev)
    except Exception as e:
        print("Run from notebook 06 after cells 0,1,2, or fix env. Error:", e)
        return

    for case in CASES:
        inputs = tokenizer(case["text"], return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inputs)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        all_probs = torch.sigmoid(out.logits[0]).cpu().numpy()
        attn = torch.stack([a[0] for a in out.attentions], dim=0).mean(dim=(0, 1)).cpu().numpy()
        name = case["name"]
        save_attention_with_labels(tokens, attn, OUT_DIR / f"attention_{name}.png")
        save_output_figure(all_probs, TARGET_NAMES, case["expected"], out_path=OUT_DIR / f"output_{name}.png")
        

if __name__ == "__main__":
    main()
