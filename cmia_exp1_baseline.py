# =============================================================================
# cmia_exp1_baseline.py
# Experiment 1 — CMIA Baseline: Joint vs. Alternating Training (biLSTM encoders)
# Depends on: cmia_shared.py
# =============================================================================
from cmia_shared import *


# ## 5. Experiment 1 — CMIA Baseline: Joint vs. Alternating Training
# 
# We first evaluate the base CMIA model (biLSTM encoders, full incongruity representation) under two training strategies. Joint training updates all parameters together every epoch. Alternating training separates the BERT fine-tuning phase from the non-BERT component training phase, preventing early-epoch noise from the randomly-initialised attention and MLP layers from corrupting BERT's pre-trained weights.
# 
# **Hypothesis**: Alternating training will outperform joint training, consistent with findings from the early fusion experiments.

class CMIAModel(nn.Module):
    """
    Base CMIA model with biLSTM encoders.
    h = [t ; ã ; ṽ ; t−ã ; t−ṽ]  (3840-dim) -> MLP -> 1 logit
    """
    def __init__(self, bert_model_name=BERT_MODEL, audio_input_dim=81,
                 vision_input_dim=371, lstm_hidden=LSTM_HIDDEN,
                 mlp_hidden=512, dropout=0.3):
        super().__init__()
        seq_dim = 2 * lstm_hidden
        assert seq_dim == BERT_DIM, (
            f"biLSTM output dim ({seq_dim}) must equal BERT_DIM ({BERT_DIM})"
        )
        self.bert        = AutoModel.from_pretrained(bert_model_name)
        self.audio_enc   = ModalityEncoder(audio_input_dim,  lstm_hidden)
        self.vision_enc  = ModalityEncoder(vision_input_dim, lstm_hidden)
        self.audio_attn  = CrossModalAttention(BERT_DIM, seq_dim)
        self.vision_attn = CrossModalAttention(BERT_DIM, seq_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(BERT_DIM * 5, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
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
        h = torch.cat([t, a_att, v_att, t - a_att, t - v_att], dim=1)
        return self.classifier(h).squeeze(1)

torch.manual_seed(42)
results = {}

print("=== Exp 1a: CMIA (biLSTM) — Joint Training ===")
model_joint = CMIAModel().to(DEVICE)
results["CMIA_joint"] = train_joint(
    model_joint, train_loader, test_loader, label="CMIA_joint")

print("=== Exp 1b: CMIA (biLSTM) — Alternating Training ===")
model_alt = CMIAModel().to(DEVICE)
results["CMIA_alternating"] = train_alternating(
    model_alt, train_loader, test_loader, label="CMIA_alternating")

# ── Results ───────────────────────────────────────────────────────────────
import pandas as pd
prior = {
    "Late Fusion RNN (avg, joint)": {"accuracy": 0.7899, "precision": 0.7568,
                                      "recall": 0.7458, "f1": 0.7434, "f1_macro": 0.7909},
}
def fmt(v): return f"{v:.4f}" if v is not None else "-"
rows = []
for label, m in [("Late Fusion RNN (avg, joint)", prior["Late Fusion RNN (avg, joint)"]),
                  ("Exp 1a — CMIA biLSTM, joint",        results["CMIA_joint"]),
                  ("Exp 1b — CMIA biLSTM, alternating",  results["CMIA_alternating"])]:
    rows.append({"Model": label,
                 "Test Acc":   fmt(m.get("accuracy")),
                 "Precision":  fmt(m.get("precision")),
                 "Recall":     fmt(m.get("recall")),
                 "F1 (pos)":   fmt(m.get("f1")),
                 "F1 (macro)": fmt(m.get("f1_macro"))})
print(pd.DataFrame(rows).to_string(index=False))

