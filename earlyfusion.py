import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import math

import pickle
import json
import numpy as np

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

class MultimodalSarcasmDataset(Dataset):
    """
    Each sample format:
    sample[0] = utterance string
    sample[1] = context string
    sample[2] = numpy array shape (50, 81)
    sample[3] = numpy array shape (50, 371)
    sample[4] = int label (0/1)
    """
    def __init__(self, sample_list, tokenizer, max_length):
        self.sample_list = sample_list
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.sample_list)

    def _encode_context_utterance(self, context_text, utterance_text):
        """
        example：
        context = A B C, utterance = D
        output: A B C D
        if exceed max length, then: B C D
        """
        merged_text = (context_text.strip() + " " + utterance_text.strip()).strip()

        merged_ids = self.tokenizer.encode(
            merged_text,
            add_special_tokens=False
        )

        available_text_len = self.max_length - 2
        if len(merged_ids) > available_text_len:
            merged_ids = merged_ids[-available_text_len:] 

        final_ids = [self.cls_id] + merged_ids + [self.sep_id]
        final_mask = [1] * len(final_ids)

        pad_len = self.max_length - len(final_ids)
        if pad_len > 0:
            final_ids = final_ids + [self.pad_id] * pad_len
            final_mask = final_mask + [0] * pad_len

        return torch.tensor(final_ids, dtype=torch.long), torch.tensor(final_mask, dtype=torch.long)

    def __getitem__(self, idx):
        sample_item = self.sample_list[idx]

        utterance_text = sample_item[0]
        context_text = sample_item[1]
        audio_feat = sample_item[2]   # (50, 81)
        vision_feat = sample_item[3]  # (50, 371)
        binary_label = sample_item[4]

        input_ids, attention_mask = self._encode_context_utterance(
            context_text=context_text,
            utterance_text=utterance_text
        )

        seq_feature = np.concatenate([audio_feat, vision_feat], axis=1)  # (50, 452)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_feature": torch.tensor(seq_feature, dtype=torch.float32),
            "label": torch.tensor(binary_label, dtype=torch.float32)
        }
    
class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim=452,
        hidden_dim=256,
        encoder_type="lstm",
        num_layers=1,
        dropout_prob=0.1
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        self.hidden_dim = hidden_dim

        if self.encoder_type == "rnn":
            self.encoder = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_prob if num_layers > 1 else 0.0,
                bidirectional=False
            )

        elif self.encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_prob if num_layers > 1 else 0.0,
                bidirectional=False
            )

        else:
            raise ValueError("encoder_type must be one of: rnn, lstm, transformer")

    def forward(self, seq_feature):
        """
        seq_feature: (batch_size, 50, 452)
        return: (batch_size, hidden_dim)
        """
        if self.encoder_type == "rnn":
            output_seq, hidden_state = self.encoder(seq_feature)
            final_feature = hidden_state[-1]  # (batch_size, hidden_dim)
            return final_feature

        elif self.encoder_type == "lstm":
            output_seq, (hidden_state, cell_state) = self.encoder(seq_feature)
            final_feature = hidden_state[-1]  # (batch_size, hidden_dim)
            return final_feature

        elif self.encoder_type == "transformer":
            projected_seq = self.input_projection(seq_feature)   # (batch_size, 50, hidden_dim)
            encoded_seq = self.encoder(projected_seq)      # (batch_size, 50, hidden_dim)
            final_feature = encoded_seq.mean(dim=1)    

class BertSeqFusionModel(nn.Module):
    def __init__(
        self,
        bert_name="bert-base-uncased",
        seq_input_dim=452,
        seq_hidden_dim=256,
        seq_encoder_type="lstm",
        seq_num_layers=1,
        seq_dropout_prob=0.1,
        mlp_hidden_dim=256,
        mlp_dropout_prob=0.2
    ):
        super().__init__()

        self.bert_model = AutoModel.from_pretrained(bert_name)
        bert_hidden_dim = self.bert_model.config.hidden_size

        self.seq_encoder = SequenceEncoder(
            input_dim=seq_input_dim,
            hidden_dim=seq_hidden_dim,
            encoder_type=seq_encoder_type,
            num_layers=seq_num_layers,
            dropout_prob=seq_dropout_prob
        )

        fusion_dim = bert_hidden_dim + seq_hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout_prob),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask, seq_feature):
        """
        """
        bert_outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feature = bert_outputs.last_hidden_state[:, 0, :]  # CLS from last hidden state

        seq_feature_encoded = self.seq_encoder(seq_feature)

        fused_feature = torch.cat([text_feature, seq_feature_encoded], dim=1)
        logits = self.classifier(fused_feature).squeeze(1)

        return logits
    
def set_requires_grad_for_module(module_obj, requires_grad_flag):
    for module_param in module_obj.parameters():
        module_param.requires_grad = requires_grad_flag

def configure_training_mode(model_obj, training_mode):
    """
    training_mode:
        "joint"       -> train: bert + seq + classifier 
        "bert_only"   -> train:  bert + classifier
        "seq_only"    -> train:  seq + classifier
    """
    if training_mode == "joint":
        set_requires_grad_for_module(model_obj.bert_model, True)
        set_requires_grad_for_module(model_obj.seq_encoder, True)
        set_requires_grad_for_module(model_obj.classifier, True)

    elif training_mode == "bert_only":
        set_requires_grad_for_module(model_obj.bert_model, True)
        set_requires_grad_for_module(model_obj.seq_encoder, False)
        set_requires_grad_for_module(model_obj.classifier, True)

    elif training_mode == "seq_only":
        set_requires_grad_for_module(model_obj.bert_model, False)
        set_requires_grad_for_module(model_obj.seq_encoder, True)
        set_requires_grad_for_module(model_obj.classifier, True)

    else:
        raise ValueError("training_mode must be one of: joint, bert_only, seq_only")


def get_epoch_training_mode(
    epoch_index,
    alternating_training=False,
    seq_phase_epochs=2,
    bert_phase_epochs=2
):
    """
    alternating_training = False:
        joint mode for all epoch

    alternating_training = True:
        example: seq_phase_epochs=2, bert_phase_epochs=2
        epoch 1-2: seq_only
        epoch 3-4: bert_only
        epoch 5-6: seq_only
        epoch 7-8: bert_only
        ...
    """
    if not alternating_training:
        return "joint"

    cycle_len = seq_phase_epochs + bert_phase_epochs
    epoch_pos = epoch_index % cycle_len

    if epoch_pos < seq_phase_epochs:
        return "seq_only"
    else:
        return "bert_only"
    
def compute_binary_accuracy(logits_tensor, labels_tensor):
    probability_tensor = torch.sigmoid(logits_tensor)
    prediction_tensor = (probability_tensor >= 0.5).float()
    correct_count = (prediction_tensor == labels_tensor).sum().item()
    total_count = labels_tensor.size(0)
    return correct_count, total_count


def run_one_epoch_train(model_obj, loader_obj, optimizer_obj, loss_fn, device_obj):
    model_obj.train()

    total_loss_sum = 0.0
    total_correct_sum = 0
    total_sample_sum = 0

    for batch_dict in loader_obj:
        batch_input_ids = batch_dict["input_ids"].to(device_obj)
        batch_attention_mask = batch_dict["attention_mask"].to(device_obj)
        batch_seq_feature = batch_dict["seq_feature"].to(device_obj)
        batch_label = batch_dict["label"].to(device_obj)

        optimizer_obj.zero_grad()

        batch_logits = model_obj(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            seq_feature=batch_seq_feature
        )

        batch_loss = loss_fn(batch_logits, batch_label)
        batch_loss.backward()
        optimizer_obj.step()

        batch_size_now = batch_label.size(0)
        total_loss_sum += batch_loss.item() * batch_size_now

        correct_count, total_count = compute_binary_accuracy(batch_logits, batch_label)
        total_correct_sum += correct_count
        total_sample_sum += total_count

    avg_loss = total_loss_sum / total_sample_sum
    avg_acc = total_correct_sum / total_sample_sum
    return avg_loss, avg_acc


@torch.no_grad()
def run_one_epoch_eval(model_obj, loader_obj, loss_fn, device_obj):
    model_obj.eval()

    total_loss_sum = 0.0
    total_correct_sum = 0
    total_sample_sum = 0

    for batch_dict in loader_obj:
        batch_input_ids = batch_dict["input_ids"].to(device_obj)
        batch_attention_mask = batch_dict["attention_mask"].to(device_obj)
        batch_seq_feature = batch_dict["seq_feature"].to(device_obj)
        batch_label = batch_dict["label"].to(device_obj)

        batch_logits = model_obj(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            seq_feature=batch_seq_feature
        )

        batch_loss = loss_fn(batch_logits, batch_label)

        batch_size_now = batch_label.size(0)
        total_loss_sum += batch_loss.item() * batch_size_now

        correct_count, total_count = compute_binary_accuracy(batch_logits, batch_label)
        total_correct_sum += correct_count
        total_sample_sum += total_count

    avg_loss = total_loss_sum / total_sample_sum
    avg_acc = total_correct_sum / total_sample_sum
    return avg_loss, avg_acc

def train_experiment2(
    train_valid_samples,
    test_samples,
    bert_name="bert-base-uncased",
    seq_encoder_type="lstm",   # "rnn", "lstm"
    max_length=128,
    batch_size=8,
    num_epochs=10,
    learning_rate=2e-6,
    seq_hidden_dim=256,
    seq_num_layers=1,
    seq_dropout_prob=0.1,
    mlp_hidden_dim=256,
    mlp_dropout_prob=0.2,
    alternating_training=False,
    seq_phase_epochs=2,
    bert_phase_epochs=2,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(bert_name)

    train_dataset = MultimodalSarcasmDataset(
        sample_list=train_valid_samples,
        tokenizer=tokenizer,
        max_length=max_length
    )
    test_dataset = MultimodalSarcasmDataset(
        sample_list=test_samples,
        tokenizer=tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = BertSeqFusionModel(
        bert_name=bert_name,
        seq_input_dim=452,
        seq_hidden_dim=seq_hidden_dim,
        seq_encoder_type=seq_encoder_type,
        seq_num_layers=seq_num_layers,
        seq_dropout_prob=seq_dropout_prob,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_dropout_prob=mlp_dropout_prob
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate
    )

    for epoch_id in range(num_epochs):
        current_training_mode = get_epoch_training_mode(
            epoch_index=epoch_id,
            alternating_training=alternating_training,
            seq_phase_epochs=seq_phase_epochs,
            bert_phase_epochs=bert_phase_epochs
        )

        configure_training_mode(model, current_training_mode)

        train_loss, train_acc = run_one_epoch_train(
            model_obj=model,
            loader_obj=train_loader,
            optimizer_obj=optimizer,
            loss_fn=loss_fn,
            device_obj=device
        )

        test_loss, test_acc = run_one_epoch_eval(
            model_obj=model,
            loader_obj=test_loader,
            loss_fn=loss_fn,
            device_obj=device
        )

        print(
            f"Epoch {epoch_id + 1:02d}/{num_epochs:02d} | "
            f"mode={current_training_mode} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} | test_acc={test_acc:.4f}"
        )

    return model, tokenizer

print("full + LSTM")
model, tokenizer = train_experiment2(
    train_valid_samples=train_valid_samples,
    test_samples=test_samples,
    bert_name="bert-base-uncased",
    seq_encoder_type="lstm",
    max_length=128,
    batch_size=16,
    num_epochs=10,
    learning_rate=2e-5,
    alternating_training=False
)
print("============================================\n\n")

print("alternative + LSTM")
model, tokenizer = train_experiment2(
    train_valid_samples=train_valid_samples,
    test_samples=test_samples,
    bert_name="bert-base-uncased",
    seq_encoder_type="lstm",
    max_length=128,
    batch_size=8,
    num_epochs=10,
    learning_rate=2e-5,
    alternating_training=True
)
print("============================================\n\n")

print("full + rnn")
model, tokenizer = train_experiment2(
    train_valid_samples=train_valid_samples,
    test_samples=test_samples,
    bert_name="bert-base-uncased",
    seq_encoder_type="rnn",
    max_length=128,
    batch_size=16,
    num_epochs=10,
    learning_rate=2e-5,
    alternating_training=False
)
print("============================================\n\n")

print("alternative + rnn")
model, tokenizer = train_experiment2(
    train_valid_samples=train_valid_samples,
    test_samples=test_samples,
    bert_name="bert-base-uncased",
    seq_encoder_type="rnn",
    max_length=128,
    batch_size=16,
    num_epochs=10,
    learning_rate=2e-5,
    alternating_training=True
)