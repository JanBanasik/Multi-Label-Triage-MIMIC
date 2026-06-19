"""
Publication-ready three-panel figure: input features, aggregated attention, predictions.
Assumes variables: tokens (list[str]), attn_matrix (np.ndarray (seq, seq)), 
all_probs (np.ndarray length 8), expected_labels (str e.g. "NEFROLOG" or "HEMATOLOG, NEFROLOG"),
class_names (list[str] length 8).
"""

import numpy as np
import matplotlib.pyplot as plt

# Group order and token prefixes for panel (a)
GROUP_ORDER = [
    "Demographics",
    "Hematology",
    "Renal",
    "Risk / Clinical",
    "Symptoms / History",
]

# Token prefix -> group (first match wins; order matters)
TOKEN_GROUP_PREFIXES = [
    ("Demographics", ["AGE_", "SEX_"]),
    ("Hematology", [
        "HGB_", "HCT_", "PLT_", "MCV_", "WBC_",
        "COMPOSITE_ANEMIA", "COMPOSITE_THROMBOCYTOPENIA", "COMPOSITE_SEVERE_ANEMIA",
    ]),
    ("Renal", ["CREATININE_", "UREA_", "COMPOSITE_RENAL"]),
    ("Symptoms / History", ["SYMPTOM_", "HIST_"]),
    ("Risk / Clinical", [
        "COMPOSITE_BLEEDING", "COMPOSITE_INFECTION", "COMPOSITE_RENAL_FAILURE",
        "COMPOSITE_BLEEDING_RISK", "COMPOSITE_INFECTION_INDICATOR",
        "ALT_", "AST_", "COMPOSITE_",
    ]),
]


def _token_to_group(token):
    for group_name, prefixes in TOKEN_GROUP_PREFIXES:
        for p in prefixes:
            if token.startswith(p):
                return group_name
    return "Risk / Clinical"


def _build_feature_list(tokens):
    """Return (feature_labels, group_per_feature) in display order (group order, then token order within group)."""
    special = {"[CLS]", "[SEP]", "[PAD]"}
    by_group = {g: [] for g in GROUP_ORDER}
    for t in tokens:
        if t in special:
            continue
        g = _token_to_group(t)
        if g not in by_group:
            by_group[g] = []
        by_group[g].append(t)
    # Within each group preserve token order (first occurrence in sequence)
    seen = set()
    order_in_seq = []
    for t in tokens:
        if t in special:
            continue
        if t not in seen:
            seen.add(t)
            order_in_seq.append(t)
    # Rebuild by_group order to match order_in_seq within each group
    by_group_ordered = {g: [] for g in GROUP_ORDER}
    for t in order_in_seq:
        g = _token_to_group(t)
        by_group_ordered[g].append(t)
    # Flatten in GROUP_ORDER; track group for each feature
    feature_labels = []
    group_per_feature = []
    for g in GROUP_ORDER:
        for t in by_group_ordered.get(g, []):
            feature_labels.append(t)
            group_per_feature.append(g)
    return feature_labels, group_per_feature


def _attention_from_cls_by_feature(tokens, attn_matrix):
    """
    attn_matrix: (seq_len, seq_len). Row i = attention FROM position i TO all positions.
    Return attention FROM [CLS] (position 0) TO each non-special token, in same order as _build_feature_list.
    """
    feature_labels, _ = _build_feature_list(tokens)
    if len(feature_labels) == 0:
        return np.array([]), []
    # Map feature label -> position in sequence (first occurrence)
    token_to_pos = {}
    for i, t in enumerate(tokens):
        if t not in token_to_pos:
            token_to_pos[t] = i
    # Attention from [CLS] (position 0) to each position
    attn_from_cls = attn_matrix[0, :]
    scores = []
    for t in feature_labels:
        pos = token_to_pos.get(t, 0)
        scores.append(attn_from_cls[pos])
    return np.array(scores), feature_labels


def plot_three_panel(
    tokens,
    attn_matrix,
    all_probs,
    expected_labels,
    class_names,
    threshold=0.15,
    out_path=None,
    fig_width=7,
    font_size=8,
):
    """
    Create a single figure with panels (a) Input features, (b) Attention, (c) Predictions.
    - tokens: list of token strings (from tokenizer)
    - attn_matrix: (seq_len, seq_len), aggregated over heads (and layers if available)
    - all_probs: length-8 array of class probabilities
    - expected_labels: str, e.g. "NEFROLOG" or "HEMATOLOG, NEFROLOG, SOR"
    - class_names: list of 8 class display names
    """
    expected_set = {s.strip().upper() for s in expected_labels.split(",")}
    # Normalize class_names for comparison (e.g. TARGET_SPEC_NEFROLOG -> NEFROLOG)
    def _norm(c):
        c = c.upper().replace("TARGET_SPEC_", "").replace("TARGET_POZ", "POZ")
        return c
    class_norm = [_norm(c) for c in class_names]
    gt_mask = np.array([class_norm[i] in expected_set for i in range(len(class_names))])

    feature_labels, group_per_feature = _build_feature_list(tokens)
    attn_scores, attn_labels = _attention_from_cls_by_feature(tokens, attn_matrix)
    if len(attn_scores) == 0:
        attn_scores = np.zeros(len(feature_labels))
        attn_labels = feature_labels

    # Layout: three panels in one row
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 3.2), gridspec_kw={"width_ratios": [1.1, 1.2, 1]})
    fig.set_facecolor("white")
    ax_a, ax_b, ax_c = axes
    for ax in axes:
        ax.tick_params(axis="both", labelsize=font_size)

    # --- Panel (a): Input features (grouped text) ---
    ax_a.set_axis_off()
    ax_a.set_title("(a) Input features", fontsize=font_size + 1, fontweight="bold", pad=6)
    y_pos = 1.0
    line_height = 0.06
    group_colors = {
        "Demographics": "#4a7ba7",
        "Hematology": "#c75b4a",
        "Renal": "#6a9f5a",
        "Risk / Clinical": "#9a7b4a",
        "Symptoms / History": "#7b6a9f",
    }
    for g in GROUP_ORDER:
        feats = [f for f, gp in zip(feature_labels, group_per_feature) if gp == g]
        if not feats:
            continue
        ax_a.text(0.02, y_pos, g + ":", fontsize=font_size, fontweight="bold", color=group_colors.get(g, "black"))
        y_pos -= line_height
        for t in feats:
            ax_a.text(0.06, y_pos, "  " + t, fontsize=font_size - 1, family="monospace")
            y_pos -= line_height
        y_pos -= line_height * 0.5
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.set_xlim(0, 1)

    # --- Panel (b): 1D attention bar plot (same order as panel a) ---
    n_f = len(attn_scores)
    if n_f > 0:
        norm = attn_scores.max() if attn_scores.max() > 0 else 1.0
        attn_norm = attn_scores / norm
        colors_b = [group_colors.get(g, "#888") for g in group_per_feature]
        y_positions = np.arange(n_f)[::-1]
        bars = ax_b.barh(y_positions, attn_norm, height=0.7, color=colors_b, edgecolor="none")
        ax_b.set_yticks(y_positions)
        ax_b.set_yticklabels(attn_labels, fontsize=font_size - 1)
        ax_b.set_xlabel("Attention (norm.)", fontsize=font_size)
        ax_b.axvline(0, color="gray", linewidth=0.5)
        ax_b.set_xlim(0, 1.05)
    ax_b.set_title("(b) Aggregated attention", fontsize=font_size + 1, fontweight="bold", pad=6)

    # --- Panel (c): Horizontal bar plot of class probabilities ---
    ax_c.set_title("(c) Predicted probabilities", fontsize=font_size + 1, fontweight="bold", pad=6)
    y_pos_c = np.arange(len(class_names))[::-1]
    colors_c = ["#2d6a4f" if gt_mask[i] else "#495057" for i in range(len(class_names))]
    ax_c.barh(y_pos_c, all_probs, height=0.7, color=colors_c, edgecolor="none")
    ax_c.axvline(threshold, color="red", linestyle="--", linewidth=1, label=f"Threshold ({threshold:.0%})")
    ax_c.set_yticks(y_pos_c)
    ax_c.set_yticklabels(class_names, fontsize=font_size)
    ax_c.set_xlabel("Probability", fontsize=font_size)
    ax_c.set_xlim(0, 1.05)
    ax_c.legend(fontsize=font_size - 1, loc="lower right")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        plt.show()
    return fig
