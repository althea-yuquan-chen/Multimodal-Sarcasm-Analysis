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

class SarcasmDualTextDataset(Dataset):
    def __init__(self, sample_list):
        self.samples = sample_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        one_sample = self.samples[idx]
        utterance_text = one_sample[0]
        context_text = one_sample[1]
        label_value = one_sample[4]
        return context_text, utterance_text, label_value


# context and utterance are tokenized separately
class DualTextCollator:
    def __init__(self, tokenizer, max_length_context=128, max_length_utterance=64):
        self.tokenizer = tokenizer
        self.max_length_context = max_length_context
        self.max_length_utterance = max_length_utterance

    def __call__(self, batch):
        context_batch = [item[0] for item in batch]
        utterance_batch = [item[1] for item in batch]
        label_batch = torch.tensor([item[2] for item in batch], dtype=torch.long)

        context_tokenized = self.tokenizer(
            context_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length_context,
            return_tensors="pt"
        )

        utterance_tokenized = self.tokenizer(
            utterance_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length_utterance,
            return_tensors="pt"
        )

        return {
            "context_input_ids": context_tokenized["input_ids"],
            "context_attention_mask": context_tokenized["attention_mask"],
            "utterance_input_ids": utterance_tokenized["input_ids"],
            "utterance_attention_mask": utterance_tokenized["attention_mask"],
            "labels": label_batch
        }


# 3. Model
# concat(BERT(context), BERT(utterance)) + MLP
# shared BERT encoder
class BertConcatMLPClassifier(nn.Module):
    def __init__(self, pretrained_name="bert-base-uncased", hidden_dim=256, dropout=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        bert_hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(
        self,
        context_input_ids,
        context_attention_mask,
        utterance_input_ids,
        utterance_attention_mask
    ):
        context_output = self.bert(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        )
        utterance_output = self.bert(
            input_ids=utterance_input_ids,
            attention_mask=utterance_attention_mask
        )

        context_cls = context_output.last_hidden_state[:, 0, :]
        utterance_cls = utterance_output.last_hidden_state[:, 0, :]

        fused_feature = torch.cat([context_cls, utterance_cls], dim=1)
        logits = self.classifier(fused_feature)
        return logits


# 4. Prepare data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = SarcasmDualTextDataset(train_valid_samples)
test_dataset = SarcasmDualTextDataset(test_samples)

collator = DualTextCollator(
    tokenizer=tokenizer,
    max_length_context=128,
    max_length_utterance=64
)

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


# 5. Model / loss / optimizer
model = BertConcatMLPClassifier(
    pretrained_name="bert-base-uncased",
    hidden_dim=256,
    dropout=0.2
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-6)


# 6. One epoch
def run_one_epoch(model, data_loader, criterion, device, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss_value = 0.0
    total_correct_num = 0
    total_sample_num = 0

    for batch_data in data_loader:
        context_input_ids = batch_data["context_input_ids"].to(device)
        context_attention_mask = batch_data["context_attention_mask"].to(device)
        utterance_input_ids = batch_data["utterance_input_ids"].to(device)
        utterance_attention_mask = batch_data["utterance_attention_mask"].to(device)
        labels = batch_data["labels"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                utterance_input_ids=utterance_input_ids,
                utterance_attention_mask=utterance_attention_mask
            )
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        pred_labels = torch.argmax(logits, dim=1)

        batch_size = labels.size(0)
        total_loss_value += loss.item() * batch_size
        total_correct_num += (pred_labels == labels).sum().item()
        total_sample_num += batch_size

    avg_loss_value = total_loss_value / total_sample_num
    avg_acc_value = total_correct_num / total_sample_num
    return avg_loss_value, avg_acc_value


# 7. Training loop
num_epochs = 10

print("Experiment 3: concat(BERT(context), BERT(utterance)) + MLP")

for epoch_num in range(1, num_epochs + 1):
    train_loss, train_acc = run_one_epoch(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        device=device,
        optimizer=optimizer
    )

    test_loss, test_acc = run_one_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None
    )

    print(
        f"Epoch {epoch_num:02d} | "
        f"train loss: {train_loss:.4f} | "
        f"train acc: {train_acc:.4f} | "
        f"test loss: {test_loss:.4f} | "
        f"test acc: {test_acc:.4f}"
    )