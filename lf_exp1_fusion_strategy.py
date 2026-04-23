# =============================================================================
# lf_exp1_fusion_strategy.py
# Experiment 1 — Fusion strategy x Training mode
# Grid: {average, weighted, mlp} x {joint, sequential}
# Depends on: lf_shared.py
# =============================================================================
from lf_shared import *


# ---
# ### Experiment 1 — Fusion strategy × Training mode
# 
# **Motivation:** your early fusion results showed alternating training improved test accuracy by ~3.6% for RNN. We test whether the same holds for late fusion, and which combiner benefits most.
# 
# Grid: `{average, weighted, mlp}` × `{joint, sequential}`
# 
# - **Joint:** all parameters updated together every step
# - **Sequential:** unimodal models trained first (BERT frozen for this phase), then frozen; combiner trained alone

def make_model(fusion_strategy="weighted", cell_type="lstm"):
    """Construct a fresh LateFusionModel."""
    text_m   = TextModel()
    audio_m  = SequenceModel(input_dim=81,  cell_type=cell_type)
    vision_m = SequenceModel(input_dim=371, cell_type=cell_type)
    combiner = LateFusionCombiner(strategy=fusion_strategy)
    return LateFusionModel(text_m, audio_m, vision_m, combiner).to(DEVICE)


def train_sequential(model, train_loader, test_loader,
                     phase1_epochs=5, phase2_epochs=5, lr=2e-5, label=""):
    """
    Sequential training strategy:
      Phase 1 — train unimodal models only (combiner frozen, BERT fine-tuned)
      Phase 2 — freeze unimodal models, train combiner only
    Returns best-F1 test metrics from phase 2.
    """
    criterion = nn.BCEWithLogitsLoss()

    # ── Phase 1: train unimodal branches, freeze combiner ──────────────────
    print(f"  [{label}] Phase 1 — training unimodal models")
    for p in model.combiner.parameters():
        p.requires_grad = False

    opt1 = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-2
    )
    for epoch in range(1, phase1_epochs + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, opt1)
        print(f"    P1 Ep {epoch:02d} | Train Acc {tr_m['accuracy']:.4f}")

    # ── Phase 2: freeze unimodal, unfreeze combiner ─────────────────────────
    print(f"  [{label}] Phase 2 — training combiner only")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.combiner.parameters():
        p.requires_grad = True

    combiner_params = list(model.combiner.parameters())
    opt2 = AdamW(combiner_params, lr=1e-3, weight_decay=1e-2) if combiner_params else None
    # opt2 = AdamW(model.combiner.parameters(), lr=1e-3, weight_decay=1e-2)
    best_f1, best_metrics, best_state = 0.0, {}, None

    for epoch in range(1, phase2_epochs + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, opt2)
        te_loss, te_m = run_epoch(model, test_loader,  criterion)
        print(f"    P2 Ep {epoch:02d} | "
              f"Train Acc {tr_m['accuracy']:.4f} | "
              f"Test  Acc {te_m['accuracy']:.4f}  F1 {te_m['f1']:.4f}")
        if te_m["f1"] > best_f1:
            best_f1      = te_m["f1"]
            best_metrics = te_m
            best_state   = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"  → Best F1: {best_f1:.4f}\n")
    return best_metrics

NUM_EPOCHS   = 10
STRATEGIES   = ["average", "weighted", "mlp"]
results_exp1 = {}   # (strategy, training_mode) -> metrics dict

for strategy in STRATEGIES:
    # Joint training
    key = (strategy, "joint")
    print(f"\n=== {key} ===")
    model = make_model(fusion_strategy=strategy)
    results_exp1[key] = train_and_eval(
        model, train_loader, test_loader,
        num_epochs=NUM_EPOCHS, label=str(key)
    )

    # Sequential training
    key = (strategy, "sequential")
    print(f"\n=== {key} ===")
    model = make_model(fusion_strategy=strategy)
    results_exp1[key] = train_sequential(
        model, train_loader, test_loader,
        phase1_epochs=NUM_EPOCHS//2,
        phase2_epochs=NUM_EPOCHS//2,
        label=str(key)
    )

# ── Results table ─────────────────────────────────────────────────────────

# ── Experiment 1: fusion strategy × training mode ──────────────────────────
rows1 = []
for (strategy, mode), m in results_exp1.items():
    rows1.append({
        "Fusion Strategy": strategy,
        "Training Mode":   mode,
        "Train Acc":       "-",    # not stored; add if needed
        "Test Acc":        f"{m['accuracy']:.4f}",
        "Precision":       f"{m['precision']:.4f}",
        "Recall":          f"{m['recall']:.4f}",
        "F1":              f"{m['f1']:.4f}",
    })
df1 = pd.DataFrame(rows1)
print("=== Exp 1: Fusion Strategy × Training Mode ===")
display(df1)

# ── Experiment 2: RNN vs LSTM ──────────────────────────────────────────────
rows2 = []
for key, m in results_exp2.items():
    rows2.append({
        "Cell Type": key.split("_")[-1].upper(),
        "Test Acc":  f"{m['accuracy']:.4f}",
        "F1":        f"{m['f1']:.4f}",
    })
df2 = pd.DataFrame(rows2)
print("\n=== Exp 2: RNN vs LSTM ===")
display(df2)

# ── Experiment 3: Modality ablation ───────────────────────────────────────
rows3 = []
for condition, m in results_exp3.items():
    rows3.append({
        "Modalities Used": condition,
        "Test Acc":        f"{m['accuracy']:.4f}",
        "Precision":       f"{m['precision']:.4f}",
        "Recall":          f"{m['recall']:.4f}",
        "F1":              f"{m['f1']:.4f}",
    })
df3 = pd.DataFrame(rows3)
print("\n=== Exp 3: Modality Ablation ===")
display(df3)
