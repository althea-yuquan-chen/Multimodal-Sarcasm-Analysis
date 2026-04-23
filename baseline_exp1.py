import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW

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

class SarcasmContextDataset(Dataset):
    def __init__(self, sample_list):
        self.context_list = [x[1] for x in sample_list]
        self.label_list = [x[4] for x in sample_list]

    def __len__(self):
        return len(self.context_list)

    def __getitem__(self, idx):
        return self.context_list[idx], self.label_list[idx]
    
class ContextCollator:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        context_batch = [x[0] for x in batch]
        label_batch = torch.tensor([x[1] for x in batch], dtype=torch.long)

        tokenized = self.tokenizer(
            context_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label_batch
        }

class BertMLPClassifier(nn.Module):
    def __init__(self, pretrained_name="bert-base-uncased", hidden_dim=256, dropout=0.2, ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = bert_out.last_hidden_state[:, 0, :]   # [CLS]
        logits = self.mlp(cls_embed)
        return logits
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = SarcasmContextDataset(train_valid_samples)
test_dataset = SarcasmContextDataset(test_samples)

collator = ContextCollator(tokenizer, max_length=128)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=collator
)

model = BertMLPClassifier(
    pretrained_name="bert-base-uncased",
    hidden_dim=256,
    dropout=0.5
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-6)
criterion = nn.CrossEntropyLoss()

def run_one_epoch(model, data_loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_data in data_loader:
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        labels = batch_data["labels"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc

num_epochs = 10

for epoch_idx in range(1, num_epochs + 1):
    train_loss, train_acc = run_one_epoch(model, train_loader, optimizer=optimizer)
    test_loss, test_acc = run_one_epoch(model, test_loader, optimizer=None)

    print(
        f"Epoch {epoch_idx:02d} | "
        f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | "
        f"test loss: {test_loss:.4f} | test acc: {test_acc:.4f}"
    )