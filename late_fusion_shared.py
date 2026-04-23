# =============================================================================
# late_fusion_shared.py
# Shared imports, dataset, model definitions, data loading, and training
# utilities. Import this module before running any experiment file.
# =============================================================================


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_MODEL = "bert-base-uncased"
print(f"Device: {DEVICE}")

# ### Dataset
# Returns each modality separately (unlike early fusion which concatenates audio+vision before returning).

class LateFusionDataset(Dataset):
    """
    sample[0] = utterance string
    sample[1] = context string
    sample[2] = np.array (50, 81)   audio
    sample[3] = np.array (50, 371)  vision
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

# ### Model definitions
# `SequenceModel` accepts a `cell_type` argument so we can swap RNN ↔ LSTM cleanly.

# ── Text model ────────────────────────────────────────────────────────────────
class TextModel(nn.Module):
    """BERT [CLS] → MLP → 1 logit."""
    def __init__(self, bert_model_name=BERT_MODEL, hidden_dim=256,
                 dropout=0.3, freeze_bert=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        bert_hidden = self.bert.config.hidden_size
        # Remove LayerNorm — BERT's CLS output is already well-behaved
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask):
        out     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = out.last_hidden_state[:, 0, :]       # (B, 768)
        return self.classifier(cls_emb).squeeze(1)     # (B,)


# ── Sequence model (audio or vision) ─────────────────────────────────────────
class SequenceModel(nn.Module):
    """
    (B, T, input_dim) → RNN/LSTM → MLP → 1 logit
    cell_type: 'lstm' | 'rnn'
    """
    def __init__(self, input_dim, hidden_dim=128, mlp_hidden=64,
                 dropout=0.3, bidirectional=True, cell_type="lstm"):
        super().__init__()
        self.bidirectional = bidirectional
        self.cell_type     = cell_type
        self.input_norm    = nn.LayerNorm(input_dim)  # keep this
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden_dim,
            batch_first=True, bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, mlp_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x):
        x = self.input_norm(x)
        output = self.rnn(x)
        # LSTM returns (out, (h_n, c_n)); RNN returns (out, h_n)
        h_n = output[1][0] if self.cell_type == "lstm" else output[1]
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2*hidden)
        else:
            h = h_n[-1]                                # (B, hidden)
        return self.classifier(h).squeeze(1)           # (B,)


# ── Combiner ──────────────────────────────────────────────────────────────────
class LateFusionCombiner(nn.Module):
    """
    strategy: 'average' | 'weighted' | 'mlp'
    """
    def __init__(self, strategy="weighted"):
        super().__init__()
        self.strategy = strategy
        if strategy == "weighted":
            self.weights = nn.Parameter(torch.ones(3))
        elif strategy == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1)
            )

    def forward(self, t, a, v):
        if self.strategy == "average":
            return (t + a + v) / 3.0
        elif self.strategy == "weighted":
            w = torch.softmax(self.weights, dim=0)
            return w[0]*t + w[1]*a + w[2]*v
        elif self.strategy == "mlp":
            return self.mlp(torch.stack([t, a, v], dim=1)).squeeze(1)


# ── Full model ────────────────────────────────────────────────────────────────
class LateFusionModel(nn.Module):
    def __init__(self, text_model, audio_model, vision_model, combiner):
        super().__init__()
        self.text_model   = text_model
        self.audio_model  = audio_model
        self.vision_model = vision_model
        self.combiner     = combiner

    def forward(self, input_ids, attention_mask, audio, vision):
        t = self.text_model(input_ids, attention_mask)
        a = self.audio_model(audio)
        v = self.vision_model(vision)
        return self.combiner(t, a, v)

# ### Training utilities

import pickle
import json

with open("data/sarcasm.pkl", "rb") as f:
    data = pickle.load(f)

with open("data/sarcasm_data.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

def build_samples(split):
    split_data = data[split]

    texts = split_data["text"]
    audios = split_data["audio"]
    visions = split_data["vision"]
    ids = split_data["id"]

    samples = []

    for i in range(len(ids)):
        sample_id = ids[i]

        if isinstance(sample_id, bytes):
            sample_id = sample_id.decode()

        # 取 json 信息
        info = meta[sample_id]

        # 拼接 context
        context_list = info["context"]
        context_str = " ".join(context_list)

        sample = [
            info["utterance"],
            context_str,
            audios[i],     #(50, 81)
            visions[i],    #(50, 371)
            int(info['sarcasm'])
        ]

        samples.append(sample)

    return samples

# 3. build
train_samples = build_samples("train")
valid_samples = build_samples("valid")
train_valid_samples = train_samples + valid_samples
test_samples  = build_samples("test")

def compute_metrics(labels, preds):
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
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
    train_ds = LateFusionDataset(train_valid_samples, tokenizer)
    test_ds  = LateFusionDataset(test_samples,        tokenizer)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False))


def train_and_eval(model, train_loader, test_loader,
                   num_epochs=10, lr_bert=2e-5, lr_seq=1e-3, label=""):
    criterion = nn.BCEWithLogitsLoss()

    # Separate lr for BERT vs the lightweight sequence models + combiner
    optimizer = AdamW([
        {"params": model.text_model.bert.parameters(),       "lr": lr_bert},
        {"params": model.text_model.classifier.parameters(), "lr": lr_seq},
        {"params": model.audio_model.parameters(),           "lr": lr_seq},
        {"params": model.vision_model.parameters(),          "lr": lr_seq},
        {"params": model.combiner.parameters(),              "lr": lr_seq},
    ], weight_decay=1e-2)

    best_f1, best_metrics, best_state = 0.0, {}, None

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer, max_grad_norm=5.0)
        te_loss, te_m = run_epoch(model, test_loader,  criterion)
        print(f"  [{label}] Ep {epoch:02d} | "
              f"Train Acc {tr_m['accuracy']:.4f} | "
              f"Test  Acc {te_m['accuracy']:.4f}  F1 {te_m['f1']:.4f}")
        if te_m["f1"] > best_f1:
            best_f1      = te_m["f1"]
            best_metrics = te_m
            best_state   = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"  → Best F1: {best_f1:.4f}\n")
    return best_metrics


# Shared tokenizer & data loaders (reused across all experiments)
tokenizer    = AutoTokenizer.from_pretrained(BERT_MODEL)
train_loader, test_loader = build_loaders(tokenizer)
