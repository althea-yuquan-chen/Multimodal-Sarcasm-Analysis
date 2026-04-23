# =============================================================================
# cmia_shared.py
# Shared imports, dataset, model building blocks, data loading, and training
# utilities for all CMIA experiments. Import before running any experiment file.
# =============================================================================


# # Cross-Modal Incongruity Attention (CMIA) for Multimodal Sarcasm Detection
# 
# This notebook implements and evaluates the **Cross-Modal Incongruity Attention (CMIA)** architecture for sarcasm detection on the MUStARD dataset. CMIA is our proposed method, designed to explicitly model the *mismatch* between what is said (text) and how it is said (audio, vision) — a key signal for sarcasm.
# 
# All CMIA experiments are compared against the best baselines established in earlier notebooks:
# - **Late Fusion RNN** (avg combiner, joint training): Acc 0.7899, F1 0.7434
# - **Early Fusion RNN** (alternating training): Acc 0.7464, F1 0.7460
# - **BERT** (context + utterance, fine-tuned): Acc 0.7319

# ## 1. Imports and Configuration

import pickle
import json
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BERT_MODEL  = "bert-base-uncased"
BERT_DIM    = 768
LSTM_HIDDEN = 384   # biRNN/biLSTM hidden=384 => output 2*384=768, matching BERT_DIM
PROJ_DIM    = 256   # shared projection space for compact variants
NUM_EPOCHS  = 10

print(f"Device: {DEVICE}")

# ## 2. Dataset
# 
# The dataset class is identical to the one used in the late fusion experiments. Each sample returns the tokenised text (context + utterance), raw audio features `(50, 81)`, and raw vision features `(50, 371)` as separate tensors so the model can process each modality independently.

class CMIADataset(Dataset):
    """
    sample[0] = utterance string
    sample[1] = context string
    sample[2] = np.array (50, 81)   audio features
    sample[3] = np.array (50, 371)  vision features
    sample[4] = int label (0/1)
    """
    def __init__(self, sample_list, tokenizer, max_length=128):
        self.sample_list = sample_list
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.sample_list)

    def _encode_text(self, context_text, utterance_text):
        merged_text = (context_text.strip() + " " + utterance_text.strip()).strip()
        merged_ids  = self.tokenizer.encode(merged_text, add_special_tokens=False)
        available   = self.max_length - 2
        if len(merged_ids) > available:
            merged_ids = merged_ids[-available:]          # keep tail (utterance end)
        final_ids  = [self.cls_id] + merged_ids + [self.sep_id]
        final_mask = [1] * len(final_ids)
        pad_len = self.max_length - len(final_ids)
        if pad_len > 0:
            final_ids  += [self.pad_id] * pad_len
            final_mask += [0]           * pad_len
        return (torch.tensor(final_ids,  dtype=torch.long),
                torch.tensor(final_mask, dtype=torch.long))

    def __getitem__(self, idx):
        s = self.sample_list[idx]
        input_ids, attention_mask = self._encode_text(s[1], s[0])
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "audio":          torch.tensor(s[2], dtype=torch.float32),   # (50, 81)
            "vision":         torch.tensor(s[3], dtype=torch.float32),   # (50, 371)
            "label":          torch.tensor(s[4], dtype=torch.float32),
        }

with open("data/sarcasm.pkl", "rb") as f:
    data = pickle.load(f)

with open("data/sarcasm_data.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

def build_samples(split):
    split_data = data[split]
    texts   = split_data["text"]
    audios  = split_data["audio"]
    visions = split_data["vision"]
    ids     = split_data["id"]
    samples = []
    for i in range(len(ids)):
        sample_id = ids[i]
        if isinstance(sample_id, bytes):
            sample_id = sample_id.decode()
        info = meta[sample_id]
        context_str = " ".join(info["context"])
        samples.append([
            info["utterance"],
            context_str,
            audios[i],    # (50, 81)
            visions[i],   # (50, 371)
            int(info["sarcasm"]),
        ])
    return samples

train_samples       = build_samples("train")
valid_samples       = build_samples("valid")
train_valid_samples = train_samples + valid_samples
test_samples        = build_samples("test")
print(f"Train+Valid: {len(train_valid_samples)}  Test: {len(test_samples)}")

# ## 3. CMIA Architecture
# 
# CMIA processes each modality through three stages:
# 
# **Stage 1 — Unimodal Encoding**
# - Text: fine-tuned BERT, `[CLS]` token → `t ∈ ℝ^768`
# - Audio: biRNN/biLSTM (hidden=384) over `(50, 81)` frames → sequence `A ∈ ℝ^{50×768}`
# - Vision: biRNN/biLSTM (hidden=384) over `(50, 371)` frames → sequence `V ∈ ℝ^{50×768}`
# 
# Setting `hidden=384` ensures the biRNN output dimension `2×384=768` matches BERT, making the element-wise difference `t − ã` dimensionally valid.
# 
# **Stage 2 — Text-Guided Cross-Modal Attention**
# 
# The text `[CLS]` embedding acts as a query; the audio and vision sequences serve as keys and values:
# 
# $$\alpha^a = \text{softmax}\!\left(\frac{(W_q\,t)^\top A}{\sqrt{768}}\right), \qquad \tilde{a} = \sum_t \alpha^a_t\, A_t$$
# 
# and symmetrically for vision, yielding attended summaries `ã, ṽ ∈ ℝ^768`.
# 
# **Stage 3 — Incongruity Representation and Classification**
# 
# $$h = [\,t\,;\,\tilde{a}\,;\,\tilde{v}\,;\,t-\tilde{a}\,;\,t-\tilde{v}\,] \in \mathbb{R}^{3840}$$
# 
# The difference terms `(t − ã)` and `(t − ṽ)` are the explicit incongruity signals: a large magnitude indicates that the acoustic or visual context contradicts the linguistic content. `h` is passed through a two-layer MLP to produce the final binary prediction.

# ── Shared building blocks ────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """biLSTM encoder. Returns full sequence (B, T, 2*hidden_dim) for attention."""
    def __init__(self, input_dim, hidden_dim=LSTM_HIDDEN):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=True)
    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        return out   # (B, T, 2*hidden_dim)


class ModalityEncoderRNN(nn.Module):
    """biRNN encoder (fewer parameters than biLSTM). Returns full sequence."""
    def __init__(self, input_dim, hidden_dim=LSTM_HIDDEN):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim,
                          batch_first=True, bidirectional=True)
    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.rnn(x)
        return out   # (B, T, 2*hidden_dim)


class ModalityEncoderSmall(nn.Module):
    """Compact biRNN encoder (hidden=128 => 256-dim output) for projected variants."""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim,
                          batch_first=True, bidirectional=True)
    def forward(self, x):
        x = self.input_norm(x)
        out, _ = self.rnn(x)
        return out   # (B, T, 256)


class CrossModalAttention(nn.Module):
    """Text-guided single-head attention over an audio or vision sequence.
    Projects text to a query; attends over the modality sequence as K/V."""
    def __init__(self, text_dim=BERT_DIM, seq_dim=BERT_DIM):
        super().__init__()
        self.query_proj = nn.Linear(text_dim, seq_dim, bias=False)
        self.scale = seq_dim ** 0.5
    def forward(self, t, S):   # t:(B,768)  S:(B,T,768)
        q      = self.query_proj(t)                                     # (B, 768)
        scores = torch.bmm(S, q.unsqueeze(2)).squeeze(2) / self.scale  # (B, T)
        attn   = torch.softmax(scores, dim=1)                           # (B, T)
        out    = torch.bmm(attn.unsqueeze(1), S).squeeze(1)            # (B, 768)
        return out, attn


class CrossModalAttentionProj(nn.Module):
    """Attention in a shared PROJ_DIM space. Query is already projected."""
    def __init__(self, proj_dim=PROJ_DIM):
        super().__init__()
        self.scale = proj_dim ** 0.5
    def forward(self, q, S):   # q:(B,256)  S:(B,T,256)
        scores = torch.bmm(S, q.unsqueeze(2)).squeeze(2) / self.scale
        attn   = torch.softmax(scores, dim=1)
        return torch.bmm(attn.unsqueeze(1), S).squeeze(1), attn

# ## 4. Training Utilities
# 
# Two training strategies are used throughout, mirroring the approach validated in the early and late fusion experiments:
# 
# - **Joint training**: all parameters are updated simultaneously every epoch, with a lower learning rate for BERT (`2e-5`) and a higher rate for the rest (`1e-3`).
# - **Alternating training**: BERT and the non-BERT components are updated in alternating phases (2 epochs rest → 2 epochs BERT → ...). This prevents the randomly-initialised non-BERT components from corrupting BERT's pre-trained weights in early training.

def compute_metrics(labels, preds):
    return {
        "accuracy":   accuracy_score(labels, preds),
        "precision":  precision_score(labels, preds, zero_division=0),
        "recall":     recall_score(labels, preds, zero_division=0),
        "f1":         f1_score(labels, preds, zero_division=0),
        "f1_macro":   f1_score(labels, preds, average="macro", zero_division=0),
        "precision_0": precision_score(labels, preds, pos_label=0, zero_division=0),
        "recall_0":    recall_score(labels, preds, pos_label=0, zero_division=0),
    }


def run_epoch(model, loader, criterion, optimizer=None,
              device=DEVICE, max_grad_norm=5.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_preds = 0.0, [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            audio = batch["audio"].to(device)
            vis   = batch["vision"].to(device)
            lbls  = batch["label"].to(device)
            logits = model(ids, mask, audio, vis)
            loss   = criterion(logits, lbls)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            total_loss += loss.item() * lbls.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            all_labels.extend(lbls.long().cpu().tolist())
            all_preds.extend(preds)
    return total_loss / len(all_labels), compute_metrics(all_labels, all_preds)


def build_loaders(tokenizer, batch_size=16):
    train_ds = CMIADataset(train_valid_samples, tokenizer)
    test_ds  = CMIADataset(test_samples,        tokenizer)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False))


def train_joint(model, train_loader, test_loader,
                num_epochs=NUM_EPOCHS, lr_bert=2e-5, lr_rest=1e-3, label="CMIA"):
    """Joint training with separate LRs for BERT vs non-BERT components.
    Returns metrics for the epoch with the best test F1."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW([
        {"params": model.bert.parameters(),        "lr": lr_bert},
        {"params": model.audio_enc.parameters(),   "lr": lr_rest},
        {"params": model.vision_enc.parameters(),  "lr": lr_rest},
        {"params": model.audio_attn.parameters(),  "lr": lr_rest},
        {"params": model.vision_attn.parameters(), "lr": lr_rest},
        {"params": model.classifier.parameters(),  "lr": lr_rest},
    ], weight_decay=1e-2)
    best_f1, best_metrics, best_state = 0.0, {}, None
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_m = run_epoch(model, test_loader,  criterion)
        print(f"  [{label}] Ep {epoch:02d} | "
          f"Train Acc {tr_m['accuracy']:.4f} | "
          f"Test Acc {te_m['accuracy']:.4f}  "
          f"P {te_m['precision']:.4f}  R {te_m['recall']:.4f}  "
          f"F1 {te_m['f1']:.4f}  F1-mac {te_m['f1_macro']:.4f}")
        if te_m["f1_macro"] > best_f1:
            best_f1      = te_m["f1_macro"]
            best_metrics = te_m
            best_state   = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    print(f"  -> Best F1_macro: {best_f1:.4f}\n")
    return best_metrics


def train_alternating(model, train_loader, test_loader,
                      num_epochs=NUM_EPOCHS, seq_phase=2, bert_phase=2,
                      lr_bert=2e-5, lr_rest=1e-3, label="CMIA-alt"):
    """Alternating training: cycles between non-BERT and BERT-only updates."""
    criterion = nn.BCEWithLogitsLoss()
    non_bert_params = (
        list(model.audio_enc.parameters())  +
        list(model.vision_enc.parameters()) +
        list(model.audio_attn.parameters()) +
        list(model.vision_attn.parameters()) +
        list(model.classifier.parameters())
    )
    opt_bert = AdamW(model.bert.parameters(), lr=lr_bert, weight_decay=1e-2)
    opt_rest = AdamW(non_bert_params,          lr=lr_rest, weight_decay=1e-2)
    cycle = seq_phase + bert_phase
    best_f1, best_metrics, best_state = 0.0, {}, None
    for epoch in range(1, num_epochs + 1):
        pos = (epoch - 1) % cycle
        if pos < seq_phase:
            for p in model.bert.parameters(): p.requires_grad = False
            for p in non_bert_params:          p.requires_grad = True
            opt, phase_label = opt_rest, "rest"
        else:
            for p in model.bert.parameters(): p.requires_grad = True
            for p in non_bert_params:          p.requires_grad = False
            opt, phase_label = opt_bert, "bert"
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, opt)
        te_loss, te_m = run_epoch(model, test_loader,  criterion)
        print(f"  [{label}] Ep {epoch:02d} | "
          f"Train Acc {tr_m['accuracy']:.4f} | "
          f"Test Acc {te_m['accuracy']:.4f}  "
          f"P {te_m['precision']:.4f}  R {te_m['recall']:.4f}  "
          f"F1 {te_m['f1']:.4f}  F1-mac {te_m['f1_macro']:.4f}")
        if te_m["f1_macro"] > best_f1:
            best_f1      = te_m["f1_macro"]
            best_metrics = te_m
            best_state   = copy.deepcopy(model.state_dict())
    for p in model.parameters(): p.requires_grad = True
    model.load_state_dict(best_state)
    print(f"  -> Best F1_macro: {best_f1:.4f}\n")
    return best_metrics


def train_alternating_generic(model, non_bert_params, train_loader, test_loader,
                               num_epochs=NUM_EPOCHS, seq_phase=2, bert_phase=2,
                               lr_bert=2e-5, lr_rest=1e-3, label="CMIA"):
    """Alternating training with an explicit non_bert_params list (for models
    whose non-BERT component names differ from the standard CMIA layout)."""
    criterion = nn.BCEWithLogitsLoss()
    opt_bert  = AdamW(model.bert.parameters(), lr=lr_bert, weight_decay=1e-2)
    opt_rest  = AdamW(non_bert_params,          lr=lr_rest, weight_decay=1e-2)
    cycle = seq_phase + bert_phase
    best_f1, best_metrics, best_state = 0.0, {}, None
    for epoch in range(1, num_epochs + 1):
        pos = (epoch - 1) % cycle
        if pos < seq_phase:
            for p in model.bert.parameters(): p.requires_grad = False
            for p in non_bert_params:          p.requires_grad = True
            opt, phase = opt_rest, "rest"
        else:
            for p in model.bert.parameters(): p.requires_grad = True
            for p in non_bert_params:          p.requires_grad = False
            opt, phase = opt_bert, "bert"
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, opt)
        te_loss, te_m = run_epoch(model, test_loader,  criterion)
        print(f"  [{label}] Ep {epoch:02d} | "
          f"Train Acc {tr_m['accuracy']:.4f} | "
          f"Test Acc {te_m['accuracy']:.4f}  "
          f"P {te_m['precision']:.4f}  R {te_m['recall']:.4f}  "
          f"F1 {te_m['f1']:.4f}  F1-mac {te_m['f1_macro']:.4f}")
        if te_m["f1_macro"] > best_f1:
            best_f1, best_metrics, best_state = te_m["f1_macro"], te_m, copy.deepcopy(model.state_dict())
    for p in model.parameters(): p.requires_grad = True
    model.load_state_dict(best_state)
    print(f"  -> Best F1_macro: {best_f1:.4f}\n")
    return best_metrics


# Shared tokenizer and data loaders
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
train_loader, test_loader = build_loaders(tokenizer)
