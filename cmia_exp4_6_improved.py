# =============================================================================
# cmia_exp4_6_improved.py
# Experiment 4 — CMIA-RNN (biRNN encoders, alternating)
# Experiment 5 — CMIA-Proj (256-dim shared projection, alternating)
# Experiment 6 — CMIA-Bidir (bidirectional cross-attention, alternating)
# Depends on: cmia_shared.py
# Run after cmia_exp1_baseline.py and cmia_exp2_3_ablation.py (or hardcode prior results)
# =============================================================================
from cmia_shared import *
results = {}   # populate from prior experiments or hardcode:
# results["CMIA_joint"]           = {...}
# results["CMIA_alternating"]     = {...}
# results["CMIA_no_diff"]         = {...}
# results["CMIA_mean_pool"]       = {...}
prior = {
    "Late Fusion RNN (avg, joint)": {"accuracy": 0.7899, "precision": 0.7568,
                                      "recall": 0.7458, "f1": 0.7434, "f1_macro": 0.7909},
    "Early Fusion RNN (alt)":       {"accuracy": 0.7464, "precision": None,
                                      "recall": None,   "f1": 0.7460, "f1_macro": None},
    "BERT (ctx+utt, fine-tuned)":   {"accuracy": 0.7319, "precision": None,
                                      "recall": None,   "f1": None,   "f1_macro": None},
}


# ## 9. Improved CMIA Variants (Experiments 4–6)
# 
# The ablations confirm that both the difference terms and the cross-modal attention are meaningful. However, overall F1 remains close to the late fusion baseline. Analysis of the precision/recall trade-off reveals that CMIA variants tend toward *higher recall at the cost of precision* — the models catch more sarcasm but with more false positives. This precision-recall imbalance, combined with the small dataset (~552 training samples), limits absolute F1 gains.
# 
# Three targeted improvements are investigated:
# 
# **Exp 4 — CMIA-RNN**: Replaces the biLSTM encoders with lighter biRNN encoders. Late fusion ablations showed biRNN outperforms biLSTM by +0.042 F1, suggesting the LSTM gating mechanism overfits on this small dataset.
# 
# **Exp 5 — CMIA-Proj**: Projects all modalities into a shared 256-dim space before attention (`h ∈ ℝ^{1280}`). This reduces the MLP input from 3840 to 1280 dimensions, directly addressing the overfitting risk from the high-dimensional representation on a small dataset.
# 
# **Exp 6 — CMIA-Bidir**: Adds a second cross-attention direction — audio and vision also attend *back* to the BERT token sequence. This creates a richer bidirectional interaction: text guides what to look for in audio/vision, and audio/vision guide what to look for in the text.
# 
# All improved variants use **alternating training**, which consistently outperformed joint training in Exp 1.

class CMIARNNModel(nn.Module):
    """
    CMIA with biRNN encoders instead of biLSTM (fewer parameters, less overfitting).
    h = [t ; ã ; ṽ ; t−ã ; t−ṽ]  (3840-dim) — same layout as base CMIA.
    """
    def __init__(self, bert_model_name=BERT_MODEL, audio_input_dim=81,
                 vision_input_dim=371, rnn_hidden=LSTM_HIDDEN,
                 mlp_hidden=512, dropout=0.3):
        super().__init__()
        seq_dim = 2 * rnn_hidden
        assert seq_dim == BERT_DIM
        self.bert        = AutoModel.from_pretrained(bert_model_name)
        self.audio_enc   = ModalityEncoderRNN(audio_input_dim,  rnn_hidden)
        self.vision_enc  = ModalityEncoderRNN(vision_input_dim, rnn_hidden)
        self.audio_attn  = CrossModalAttention(BERT_DIM, seq_dim)
        self.vision_attn = CrossModalAttention(BERT_DIM, seq_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(BERT_DIM * 5, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 128),           nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, input_ids, attention_mask, audio, vision):
        t = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask).last_hidden_state[:, 0, :]
        A = self.audio_enc(audio)
        V = self.vision_enc(vision)
        a_att, _ = self.audio_attn(t, A)
        v_att, _ = self.vision_attn(t, V)
        h = torch.cat([t, a_att, v_att, t - a_att, t - v_att], dim=1)
        return self.classifier(h).squeeze(1)


class CMIAProjModel(nn.Module):
    """
    CMIA with a shared 256-dim projection space.
    All three modalities are projected to PROJ_DIM before attention.
    h = [t_proj ; ã ; ṽ ; t_proj−ã ; t_proj−ṽ]  (1280-dim) — smaller MLP, less overfitting.
    """
    def __init__(self, bert_model_name=BERT_MODEL, audio_input_dim=81,
                 vision_input_dim=371, proj_dim=PROJ_DIM,
                 mlp_hidden=256, dropout=0.4):
        super().__init__()
        self.bert        = AutoModel.from_pretrained(bert_model_name)
        self.text_proj   = nn.Linear(BERT_DIM, proj_dim, bias=False)
        self.audio_enc   = ModalityEncoderSmall(audio_input_dim)
        self.vision_enc  = ModalityEncoderSmall(vision_input_dim)
        self.audio_attn  = CrossModalAttentionProj(proj_dim)
        self.vision_attn = CrossModalAttentionProj(proj_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(proj_dim * 5, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 64),            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, input_ids, attention_mask, audio, vision):
        t      = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state[:, 0, :]
        t_proj = self.text_proj(t)
        A = self.audio_enc(audio)
        V = self.vision_enc(vision)
        a_att, _ = self.audio_attn(t_proj, A)
        v_att, _ = self.vision_attn(t_proj, V)
        h = torch.cat([t_proj, a_att, v_att, t_proj - a_att, t_proj - v_att], dim=1)
        return self.classifier(h).squeeze(1)


class CMIABidirectionalModel(nn.Module):
    """
    Bidirectional cross-modal attention.
    Direction 1 (original): text [CLS] queries audio/vision sequences -> ã, ṽ
    Direction 2 (new):      audio/vision mean queries BERT token sequence -> t_from_a, t_from_v
    h = [t ; ã ; ṽ ; t_from_a ; t_from_v ; t−ã ; t−ṽ ; t_from_a−t ; t_from_v−t]  (6912-dim)
    """
    def __init__(self, bert_model_name=BERT_MODEL, audio_input_dim=81,
                 vision_input_dim=371, rnn_hidden=LSTM_HIDDEN,
                 mlp_hidden=512, dropout=0.3):
        super().__init__()
        seq_dim = 2 * rnn_hidden
        self.bert       = AutoModel.from_pretrained(bert_model_name)
        self.audio_enc  = ModalityEncoderRNN(audio_input_dim,  rnn_hidden)
        self.vision_enc = ModalityEncoderRNN(vision_input_dim, rnn_hidden)
        self.audio_attn  = CrossModalAttention(BERT_DIM, seq_dim)
        self.vision_attn = CrossModalAttention(BERT_DIM, seq_dim)
        self.text_attn_from_audio  = CrossModalAttention(seq_dim, BERT_DIM)
        self.text_attn_from_vision = CrossModalAttention(seq_dim, BERT_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(BERT_DIM * 9, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 256),           nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
    def forward(self, input_ids, attention_mask, audio, vision):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t        = bert_out.last_hidden_state[:, 0, :]   # (B, 768)
        text_seq = bert_out.last_hidden_state             # (B, L, 768)
        A = self.audio_enc(audio)
        V = self.vision_enc(vision)
        a_att, _    = self.audio_attn(t, A)
        v_att, _    = self.vision_attn(t, V)
        a_query     = A.mean(dim=1)
        v_query     = V.mean(dim=1)
        t_from_a, _ = self.text_attn_from_audio(a_query,  text_seq)
        t_from_v, _ = self.text_attn_from_vision(v_query, text_seq)
        h = torch.cat([t, a_att, v_att, t_from_a, t_from_v,
                        t - a_att, t - v_att,
                        t_from_a - t, t_from_v - t], dim=1)
        return self.classifier(h).squeeze(1)

torch.manual_seed(42)

# Exp 4: CMIA-RNN
print("=== Exp 4: CMIA-RNN — Alternating Training ===")
model_rnn = CMIARNNModel().to(DEVICE)
results["CMIA_RNN"] = train_alternating(
    model_rnn, train_loader, test_loader, label="CMIA_RNN")

# Exp 5: CMIA-Proj
print("=== Exp 5: CMIA-Proj — Alternating Training ===")
model_proj = CMIAProjModel().to(DEVICE)
proj_non_bert = (
    list(model_proj.text_proj.parameters())   +
    list(model_proj.audio_enc.parameters())   +
    list(model_proj.vision_enc.parameters())  +
    list(model_proj.audio_attn.parameters())  +
    list(model_proj.vision_attn.parameters()) +
    list(model_proj.classifier.parameters())
)
results["CMIA_Proj"] = train_alternating_generic(
    model_proj, proj_non_bert, train_loader, test_loader, label="CMIA_Proj")

# Exp 6: CMIA-Bidir
print("=== Exp 6: CMIA-Bidir — Alternating Training ===")
model_bidir = CMIABidirectionalModel().to(DEVICE)
results["CMIA_Bidir"] = train_alternating(
    model_bidir, train_loader, test_loader, label="CMIA_Bidir")

# ## 10. Final Results
# 
# The table below presents all CMIA variants alongside the reference baselines. Models are grouped by purpose: baselines, training strategy comparison, ablations, and improved variants.
# 
# **Columns**: *Precision* and *Recall* are computed for the positive (sarcastic) class.
# *F1 (pos)* is the positive-class F1. *F1 (macro)* is the unweighted average of positive
# and negative class F1, which is a fairer summary metric given MUStARD's class imbalance
# (~60% non-sarcastic in the test set). A model that merely predicts the majority class
# achieves ~0.72 accuracy but a macro F1 well below 0.70, so macro F1 is the primary
# comparison metric.
# 
# **Reading the precision/recall columns**: Several CMIA variants achieve accuracy comparable to or exceeding late fusion, but the F1 ceiling remains similar. This is explained by a systematic precision-recall trade-off — models with higher accuracy tend to predict sarcasm more conservatively (higher precision, lower recall), while models with higher recall generate more false positives. F1 is the harmonic mean of both, so it stays bounded regardless of which direction the model leans. This behaviour reflects the inherent *text dominance* in MUStARD: the audio and vision features carry limited additional signal, so cross-modal incongruity features act as a soft corrective rather than a primary discriminator.

all_results = {**prior, **results}

final_order = [
    # Baselines
    ("Late Fusion RNN (avg, joint)", prior,   "Baseline — Late Fusion RNN (avg, joint)"),
    ("Early Fusion RNN (alt)",       prior,   "Baseline — Early Fusion RNN (alternating)"),
    ("BERT (ctx+utt, fine-tuned)",   prior,   "Baseline — BERT (ctx+utt, fine-tuned)"),
    # Exp 1: training strategy
    ("CMIA_joint",                   results, "Exp 1a — CMIA biLSTM, joint"),
    ("CMIA_alternating",             results, "Exp 1b — CMIA biLSTM, alternating"),
    # Exp 2–3: ablations
    ("CMIA_no_diff",                 results, "Exp 2  — CMIA, no diff terms (joint)"),
    ("CMIA_mean_pool",               results, "Exp 3  — CMIA, mean pool (joint)"),
    # Exp 4–6: improved variants
    ("CMIA_RNN",                     results, "Exp 4  — CMIA-RNN (alternating)"),
    ("CMIA_Proj",                    results, "Exp 5  — CMIA-Proj 256-dim (alternating)"),
    ("CMIA_Bidir",                   results, "Exp 6  — CMIA-Bidir (alternating)"),
]

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

rows = [make_row(label, src.get(key, {})) for key, src, label in final_order]
display(pd.DataFrame(rows))

# ## 11. Attention Weight Analysis (Optional)
# 
# To provide qualitative insight into what the cross-modal attention has learned, we extract the attention weight distributions for the best-performing CMIA model on a sample of test utterances. For each sample, we report the peak attention timestep and the entropy of the distribution: low entropy indicates sharp, focused attention (the model has identified a specific acoustic or visual moment as relevant); high entropy indicates diffuse attention across the sequence (consistent with audio/vision providing little discriminative signal).

# Use the best CMIA model (alternating) for attention inspection
model_alt.eval()
sample_batch = next(iter(test_loader))

with torch.no_grad():
    ids   = sample_batch["input_ids"].to(DEVICE)
    mask  = sample_batch["attention_mask"].to(DEVICE)
    audio = sample_batch["audio"].to(DEVICE)
    vis   = sample_batch["vision"].to(DEVICE)
    lbls  = sample_batch["label"]

    bert_out = model_alt.bert(input_ids=ids, attention_mask=mask)
    t = bert_out.last_hidden_state[:, 0, :]
    A = model_alt.audio_enc(audio)
    V = model_alt.vision_enc(vis)
    _, audio_weights  = model_alt.audio_attn(t, A)   # (B, 50)
    _, vision_weights = model_alt.vision_attn(t, V)  # (B, 50)

print("Audio attention weights — first 5 test samples (50 timesteps each):")
for i in range(min(5, audio_weights.size(0))):
    w    = audio_weights[i].cpu().numpy()
    peak = w.argmax()
    ent  = -(w * np.log(w + 1e-9)).sum()
    print(f"  Sample {i} (label={int(lbls[i])}) — "
          f"peak at t={peak}, weight={w[peak]:.4f}, entropy={ent:.2f}")

print()
print("Vision attention weights — first 5 test samples:")
for i in range(min(5, vision_weights.size(0))):
    w    = vision_weights[i].cpu().numpy()
    peak = w.argmax()
    ent  = -(w * np.log(w + 1e-9)).sum()
    print(f"  Sample {i} (label={int(lbls[i])}) — "
          f"peak at t={peak}, weight={w[peak]:.4f}, entropy={ent:.2f}")
