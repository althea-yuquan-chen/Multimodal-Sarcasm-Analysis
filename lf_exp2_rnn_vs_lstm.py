# =============================================================================
# lf_exp2_rnn_vs_lstm.py
# Experiment 2 — RNN vs LSTM sequence encoders (weighted combiner, joint training)
# Depends on: lf_shared.py
# =============================================================================
from lf_shared import *

def make_model(fusion_strategy="weighted", cell_type="lstm"):
    """Construct a fresh LateFusionModel."""
    text_m   = TextModel()
    audio_m  = SequenceModel(input_dim=81,  cell_type=cell_type)
    vision_m = SequenceModel(input_dim=371, cell_type=cell_type)
    combiner = LateFusionCombiner(strategy=fusion_strategy)
    return LateFusionModel(text_m, audio_m, vision_m, combiner).to(DEVICE)


# ---
# ### Experiment 2 — RNN vs LSTM for sequence encoders
# 
# **Motivation:** your early fusion results showed BERT+RNN outperformed BERT+LSTM under alternating training (74.64% vs 71.74%). We check if this pattern transfers to the late fusion audio/vision branches.

results_exp2 = {}

for cell_type in ["lstm", "rnn"]:
    key = f"weighted_joint_{cell_type}"
    print(f"\n=== {key} ===")
    model = make_model(fusion_strategy="weighted", cell_type=cell_type)
    results_exp2[key] = train_and_eval(
        model, train_loader, test_loader,
        num_epochs=NUM_EPOCHS, label=key
    )

# ── Results table ─────────────────────────────────────────────────────────
import pandas as pd
rows = []
for key, m in results_exp2.items():
    rows.append({
        "Config":     key,
        "Test Acc":   f"{m['accuracy']:.4f}",
        "Precision":  f"{m['precision']:.4f}",
        "Recall":     f"{m['recall']:.4f}",
        "F1 (pos)":   f"{m['f1']:.4f}",
        "F1 (macro)": f"{m['f1_macro']:.4f}" if m.get("f1_macro") else "-",
    })
print(pd.DataFrame(rows).to_string(index=False))

