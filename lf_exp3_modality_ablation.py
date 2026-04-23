# =============================================================================
# lf_exp3_modality_ablation.py
# Experiment 3 — Modality ablation (text / audio / vision / combinations)
# Depends on: lf_shared.py
# =============================================================================
from lf_shared import *

def make_model(fusion_strategy="weighted", cell_type="rnn"):
    """Construct a fresh LateFusionModel."""
    text_m   = TextModel()
    audio_m  = SequenceModel(input_dim=81,  cell_type=cell_type)
    vision_m = SequenceModel(input_dim=371, cell_type=cell_type)
    combiner = LateFusionCombiner(strategy=fusion_strategy)
    return LateFusionModel(text_m, audio_m, vision_m, combiner).to(DEVICE)


# ---
# ### Experiment 3 — Modality ablation
# 
# **Motivation:** your best text-only baseline already hits 73.19%. Ablation will reveal whether audio/vision actually contribute anything, or if BERT is doing all the work.
# 
# We zero-out the logit of the excluded modality at inference time by replacing its model with a dummy that always returns 0.

class ZeroModel(nn.Module):
    """Drop-in replacement that always outputs a zero logit — used for ablation."""
    def forward(self, *args, **kwargs):
        # infer batch size from the first argument
        x = args[0]
        return torch.zeros(x.size(0), device=x.device)


def run_ablation(full_model, test_loader, ablate):
    """
    Evaluate full_model with one or more modalities zeroed out.
    ablate: list of modalities to zero, e.g. ['audio', 'vision']
    """
    # Temporarily swap out the ablated branch
    saved = {}
    if "text" in ablate:
        saved["text"]   = full_model.text_model
        full_model.text_model   = ZeroModel().to(DEVICE)
    if "audio" in ablate:
        saved["audio"]  = full_model.audio_model
        full_model.audio_model  = ZeroModel().to(DEVICE)
    if "vision" in ablate:
        saved["vision"] = full_model.vision_model
        full_model.vision_model = ZeroModel().to(DEVICE)

    _, metrics = run_epoch(full_model, test_loader, nn.BCEWithLogitsLoss())

    # Restore
    for k, v in saved.items():
        setattr(full_model, f"{k}_model", v)

    return metrics

# Train a single full model with the best strategy found in Exp 1
# (update fusion_strategy here if Exp 1 reveals a different winner)
print("Training full model for ablation study...")
ablation_model = make_model(fusion_strategy="weighted")
train_and_eval(ablation_model, train_loader, test_loader,
               num_epochs=NUM_EPOCHS, label="ablation_full")

# Run ablation conditions
ablation_conditions = {
    "text only":          ["audio", "vision"],
    "audio only":         ["text",  "vision"],
    "vision only":        ["text",  "audio"],
    "text + audio":       ["vision"],
    "text + vision":      ["audio"],
    "audio + vision":     ["text"],
    "all modalities":     [],
}

results_exp3 = {}
for condition, ablate in ablation_conditions.items():
    results_exp3[condition] = run_ablation(ablation_model, test_loader, ablate)
    print(f"  {condition:20s} | "
          f"Acc {results_exp3[condition]['accuracy']:.4f}  "
          f"F1  {results_exp3[condition]['f1']:.4f}")

# ── Combiner weights ──────────────────────────────────────────────────────

w = torch.softmax(ablation_model.combiner.weights, dim=0)
print(f"text={w[0]:.4f}  audio={w[1]:.4f}  vision={w[2]:.4f}")

# ── Per-sample logit analysis ─────────────────────────────────────────────

ablation_model.eval()
all_text, all_audio, all_vision = [], [], []

with torch.no_grad():
    for batch in test_loader:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        audio = batch["audio"].to(DEVICE)
        vis   = batch["vision"].to(DEVICE)

        all_text.append(ablation_model.text_model(ids, mask))
        all_audio.append(ablation_model.audio_model(audio))
        all_vision.append(ablation_model.vision_model(vis))

text_logits   = torch.cat(all_text).cpu()
audio_logits  = torch.cat(all_audio).cpu()
vision_logits = torch.cat(all_vision).cpu()

print(f"Text  — mean: {text_logits.mean():.4f}  std: {text_logits.std():.4f}")
print(f"Audio — mean: {audio_logits.mean():.4f}  std: {audio_logits.std():.4f}")
print(f"Vision— mean: {vision_logits.mean():.4f}  std: {vision_logits.std():.4f}")

# ── Gradient norm diagnostic (run if needed) ──────────────────────────────

total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"grad norm: {total_norm:.4f}")  # if consistently >> 1.0, clipping is needed
