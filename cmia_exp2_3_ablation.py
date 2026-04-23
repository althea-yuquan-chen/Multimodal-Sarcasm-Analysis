# =============================================================================
# cmia_exp2_3_ablation.py
# Experiment 2 — Ablation: Incongruity difference features
# Experiment 3 — Ablation: Cross-modal attention vs. mean pooling
# Depends on: cmia_shared.py  (run cmia_exp1_baseline.py first to populate results{})
# =============================================================================
from cmia_shared import *
results = {}   # populate from exp1 or hardcode prior run values below
# results["CMIA_joint"]       = {"accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "f1_macro": ...}
# results["CMIA_alternating"] = {"accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "f1_macro": ...}


# ## 6. Experiment 2 — Ablation: Do the Incongruity Difference Features Help?
# 
# The key architectural claim of CMIA is that the difference terms `(t − ã)` and `(t − ṽ)` provide an explicit incongruity signal. To verify this, we compare the full CMIA representation `[t; ã; ṽ; t−ã; t−ṽ]` against a version that omits the difference terms, using only `[t; ã; ṽ]`.
# 
# **Hypothesis**: Removing the difference terms will reduce F1, confirming that the explicit cross-modal mismatch features carry useful discriminative information beyond simple attended fusion.

class CMIAModelNoDiff(nn.Module):
    """
    CMIA without difference terms.
    h = [t ; ã ; ṽ]  (2304-dim) — no explicit incongruity signal.
    """
    def __init__(self, bert_model_name=BERT_MODEL, audio_input_dim=81,
                 vision_input_dim=371, lstm_hidden=LSTM_HIDDEN,
                 mlp_hidden=512, dropout=0.3):
        super().__init__()
        seq_dim = 2 * lstm_hidden
        self.bert        = AutoModel.from_pretrained(bert_model_name)
        self.audio_enc   = ModalityEncoder(audio_input_dim,  lstm_hidden)
        self.vision_enc  = ModalityEncoder(vision_input_dim, lstm_hidden)
        self.audio_attn  = CrossModalAttention(BERT_DIM, seq_dim)
        self.vision_attn = CrossModalAttention(BERT_DIM, seq_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(BERT_DIM * 3, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 128),           nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, input_ids, attention_mask, audio, vision):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t  = bert_out.last_hidden_state[:, 0, :]
        A  = self.audio_enc(audio)
        V  = self.vision_enc(vision)
        a_att, _ = self.audio_attn(t, A)
        v_att, _ = self.vision_attn(t, V)
        h = torch.cat([t, a_att, v_att], dim=1)   # no diff terms
        return self.classifier(h).squeeze(1)

torch.manual_seed(42)

print("=== Exp 2: CMIA (no diff terms) — Joint Training ===")
model_nodiff = CMIAModelNoDiff().to(DEVICE)

criterion_nd = nn.BCEWithLogitsLoss()
opt_nd = AdamW([
    {"params": model_nodiff.bert.parameters(),        "lr": 2e-5},
    {"params": model_nodiff.audio_enc.parameters(),   "lr": 1e-3},
    {"params": model_nodiff.vision_enc.parameters(),  "lr": 1e-3},
    {"params": model_nodiff.audio_attn.parameters(),  "lr": 1e-3},
    {"params": model_nodiff.vision_attn.parameters(), "lr": 1e-3},
    {"params": model_nodiff.classifier.parameters(),  "lr": 1e-3},
], weight_decay=1e-2)

label = "CMIA_no_diff"
best_f1_nd, best_m_nd, best_state_nd = 0.0, {}, None
for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_m = run_epoch(model_nodiff, train_loader, criterion_nd, opt_nd)
    te_loss, te_m = run_epoch(model_nodiff, test_loader,  criterion_nd)
    print(f"  [{label}] Ep {epoch:02d} | "
      f"Train Acc {tr_m['accuracy']:.4f} | "
      f"Test Acc {te_m['accuracy']:.4f}  "
      f"P {te_m['precision']:.4f}  R {te_m['recall']:.4f}  "
      f"F1 {te_m['f1']:.4f}  F1-mac {te_m['f1_macro']:.4f}")
    if te_m["f1_macro"] > best_f1_nd:
        best_f1_nd, best_m_nd, best_state_nd = te_m["f1_macro"], te_m, copy.deepcopy(model_nodiff.state_dict())

model_nodiff.load_state_dict(best_state_nd)
results["CMIA_no_diff"] = best_m_nd
print(f"  -> Best F1_macro: {best_f1_nd:.4f}\n")

# ## 7. Experiment 3 — Ablation: Does Text-Guided Attention Help over Mean Pooling?
# 
# The second architectural claim is that using the text `[CLS]` vector to *selectively attend* over the audio/vision sequence is better than simply averaging it. This ablation replaces the cross-modal attention with mean pooling over the biLSTM output, while keeping the incongruity difference terms.
# 
# **Hypothesis**: Mean pooling will underperform text-guided attention, confirming that the text-conditional selection of relevant acoustic/visual frames is a meaningful inductive bias.

class CMIAModelMeanPool(nn.Module):
    """
    CMIA with mean pooling instead of text-guided cross-modal attention.
    h = [t ; mean(A) ; mean(V) ; t−mean(A) ; t−mean(V)]  (3840-dim).
    """
    def __init__(self, bert_model_name=BERT_MODEL, audio_input_dim=81,
                 vision_input_dim=371, lstm_hidden=LSTM_HIDDEN,
                 mlp_hidden=512, dropout=0.3):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(bert_model_name)
        self.audio_enc  = ModalityEncoder(audio_input_dim,  lstm_hidden)
        self.vision_enc = ModalityEncoder(vision_input_dim, lstm_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(BERT_DIM * 5, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 128),           nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, input_ids, attention_mask, audio, vision):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t  = bert_out.last_hidden_state[:, 0, :]
        A  = self.audio_enc(audio)
        V  = self.vision_enc(vision)
        a_pool = A.mean(dim=1)
        v_pool = V.mean(dim=1)
        h = torch.cat([t, a_pool, v_pool, t - a_pool, t - v_pool], dim=1)
        return self.classifier(h).squeeze(1)

torch.manual_seed(42)

print("=== Exp 3: CMIA (mean pool) — Joint Training ===")
model_pool = CMIAModelMeanPool().to(DEVICE)

criterion_mp = nn.BCEWithLogitsLoss()
opt_mp = AdamW([
    {"params": model_pool.bert.parameters(),       "lr": 2e-5},
    {"params": model_pool.audio_enc.parameters(),  "lr": 1e-3},
    {"params": model_pool.vision_enc.parameters(), "lr": 1e-3},
    {"params": model_pool.classifier.parameters(), "lr": 1e-3},
], weight_decay=1e-2)

best_f1_mp, best_m_mp, best_state_mp = 0.0, {}, None
for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_m = run_epoch(model_pool, train_loader, criterion_mp, opt_mp)
    te_loss, te_m = run_epoch(model_pool, test_loader,  criterion_mp)
    print(f"  [CMIA_pool] Ep {epoch:02d} | "
      f"Train Acc {tr_m['accuracy']:.4f} | "
      f"Test Acc {te_m['accuracy']:.4f}  "
      f"P {te_m['precision']:.4f}  R {te_m['recall']:.4f}  "
      f"F1 {te_m['f1']:.4f}  F1-mac {te_m['f1_macro']:.4f}")
    if te_m["f1_macro"] > best_f1_mp:
        best_f1_mp, best_m_mp, best_state_mp = te_m["f1_macro"], te_m, copy.deepcopy(model_pool.state_dict())

model_pool.load_state_dict(best_state_mp)
results["CMIA_mean_pool"] = best_m_mp
print(f"  -> Best F1_macro: {best_f1_mp:.4f}\n")

# ## 8. Ablation Summary (Experiments 1–3)
# 
# The table below summarises the results from the three base experiments before we proceed to improved variants. Key comparisons:
# 
# - **Exp 1**: Joint vs. alternating training — does curriculum-style training help?
# - **Exp 2**: Full incongruity `[t; ã; ṽ; t−ã; t−ṽ]` vs. no diff terms `[t; ã; ṽ]` — do difference features add value?
# - **Exp 3**: Text-guided attention vs. mean pooling — does the selective attention mechanism matter?

prior = {
    "Late Fusion RNN (avg, joint)": {"accuracy": 0.7899, "precision": 0.7568, "recall": 0.7458, "f1": 0.7434, "f1_macro": 0.7909,},
    "Early Fusion RNN (alt)":       {"accuracy": 0.7464, "precision": None,   "recall": None,   "f1": 0.7460, "f1_macro": None},
    "BERT (ctx+utt, fine-tuned)":   {"accuracy": 0.7319, "precision": None,   "recall": None,   "f1": None , "f1_macro": None },
}

def fmt(v): return f"{v:.4f}" if v is not None else "-"

def make_row(label, m):
    return {
        "Model / Variant": label,
        "Test Acc":   fmt(m.get("accuracy")),
        "Precision":  fmt(m.get("precision")),
        "Recall":     fmt(m.get("recall")),
        "F1 (pos)":   fmt(m.get("f1")),
        "F1 (macro)": fmt(m.get("f1_macro")),
    }

ablation_order = [
    ("Late Fusion RNN (avg, joint)", prior,   "Baseline"),
    ("CMIA_joint",                   results, "Exp 1a — biLSTM, joint"),
    ("CMIA_alternating",             results, "Exp 1b — biLSTM, alternating"),
    ("CMIA_no_diff",                 results, "Exp 2  — no diff terms (joint)"),
    ("CMIA_mean_pool",               results, "Exp 3  — mean pool (joint)"),
]

rows = [make_row(label, src.get(key, {})) for key, src, label in ablation_order]
display(pd.DataFrame(rows))
