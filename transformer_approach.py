"""
From-scratch transformer for SPR 2026 Mammography BI-RADS Classification.

Pipeline:
  1. Setup & config
  2. Data loading
  3. Preprocessing (clean_text + dedup)
  4. Custom BPE tokenizer (trained via basic.py)
  5. Dataset / DataLoader
  6. Model
  7. Training & evaluation helpers
  8. StratifiedKFold CV + fold-model ensemble submission
"""

# ── 1. Setup ─────────────────────────────────────────────────────────────────

import re
import sys
from contextlib import nullcontext

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

# --- Paths ---
MODEL_DIR = "/kaggle/input/models/lokeshgautham/smallgpt/pytorch/default/1"
COMP_DIR = "/kaggle/input/competitions/spr-2026-mammography-report-classification"

sys.path.insert(0, MODEL_DIR)

from basic import BasicTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

USE_AMP = DEVICE_TYPE == "cuda"
AMP_CTX = torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.bfloat16) if USE_AMP else nullcontext()

COMPILE_MODEL = True

# ── Config ───────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
N_SPLITS = 5
NUM_CLASSES = 7  # BI-RADS 0-6

# --- Tokenizer ---
TOKENIZER_PATH = f"{MODEL_DIR}/mammo_tokenizer_4096.model"

# --- Model architecture ---
block_size = 64

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
# Encoder (pretrained by MLM) uses a fraction of LEARNING_RATE; cls_* uses full LR.
LR_ENCODER_MULT = 0.25
WEIGHT_DECAY = 0.1
EPOCHS = 20
EARLY_STOP_PATIENCE = 4
USE_BALANCED_SAMPLER = True

# --- Paths ---
TRAIN_PATH = f"{COMP_DIR}/train.csv"
TEST_PATH = f"{COMP_DIR}/test.csv"

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ── 2. Data loading ──────────────────────────────────────────────────────────

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")


# ── 3. Preprocessing ─────────────────────────────────────────────────────────

_whitespace_re = re.compile(r"[ \t]+")
_newlines_re = re.compile(r"\n{2,}")

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _whitespace_re.sub(" ", s)
    s = _newlines_re.sub("\n", s)
    return s


train_df["text"] = train_df["report"].apply(clean_text)
test_df["text"] = test_df["report"].apply(clean_text)

before = len(train_df)
train_df = train_df.drop_duplicates(subset=["text", "target"]).reset_index(drop=True)
print(f"Dedup: {before} -> {len(train_df)} rows ({before - len(train_df)} duplicates removed)")


# ── 4. Load custom BPE tokenizer ─────────────────────────────────────────────

tokenizer = BasicTokenizer()
tokenizer.load(TOKENIZER_PATH)

VOCAB_SIZE = len(tokenizer.vocab)
PAD_ID = tokenizer.pad_id
MASK_ID = tokenizer.mask_id

print(f"Tokenizer loaded: vocab_size={VOCAB_SIZE}, pad_id={PAD_ID}, mask_id={MASK_ID}")


# ── 5. Dataset ───────────────────────────────────────────────────────────────

class MammoDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, attn_mask = tokenizer.encode_with_padding(self.texts[idx], block_size)
        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_train_loader(dataset, labels, batch_size, sampler_seed: int):
    """Training loader with per-class-balanced sampling (oversamples rare BI-RADS)."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    w_class = 1.0 / counts
    sample_w = torch.as_tensor(w_class[labels], dtype=torch.double)
    gen = torch.Generator()
    gen.manual_seed(int(sampler_seed))
    sampler = WeightedRandomSampler(
        sample_w,
        num_samples=len(dataset),
        replacement=True,
        generator=gen,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
    )


# ── 6. Model ─────────────────────────────────────────────────────────────────

from model import SmallGPT, SmallGPTConfig


def build_model():
    config = SmallGPTConfig(
        block_size=block_size,
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        n_layer=5,
        n_embd=192,
        n_head=4,
        dropout=0.2,
    )
    return SmallGPT(config)


# ── 6b. MLM Pre-training ─────────────────────────────────────────────────────

MLM_EPOCHS = 100
MLM_LR = 5e-4
MLM_BATCH_SIZE = 64
MASK_PROB = 0.20

class MLMDataset(Dataset):
    """Tokenizes texts and applies random masking for MLM."""
    def __init__(self, texts):
        self.encoded = [tokenizer.encode_with_padding(t, block_size) for t in texts]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        ids, attn_mask = self.encoded[idx]
        ids = torch.tensor(ids, dtype=torch.long)
        attn_mask = torch.tensor(attn_mask, dtype=torch.long)

        # Only mask real tokens (where attn_mask == 1)
        labels = ids.clone()
        real_mask = attn_mask.bool()
        rand = torch.rand(ids.shape)
        mask_positions = (rand < MASK_PROB) & real_mask

        # 80% replace with [MASK], 10% random token, 10% keep original
        replace_mask = torch.rand(ids.shape) < 0.8
        random_mask = (torch.rand(ids.shape) < 0.5) & ~replace_mask
        ids[mask_positions & replace_mask] = MASK_ID
        ids[mask_positions & random_mask] = torch.randint(0, VOCAB_SIZE, ids.shape)[mask_positions & random_mask]

        # Only compute loss on masked positions
        labels[~mask_positions] = -100

        return {"input_ids": ids, "attention_mask": attn_mask, "labels": labels}


def pretrain_mlm(all_texts):
    """Pre-train the transformer encoder with MLM on all reports."""
    print("\n" + "="*60)
    print("MLM Pre-training")
    print("="*60)

    model = build_model().to(DEVICE)

    ds = MLMDataset(all_texts)
    loader = DataLoader(ds, batch_size=MLM_BATCH_SIZE, shuffle=True)

    optimizer = model.configure_optimizers(
        WEIGHT_DECAY, MLM_LR, (0.9, 0.95), DEVICE_TYPE
    )
    total_steps = len(loader) * MLM_EPOCHS
    scheduler = OneCycleLR(
        optimizer, max_lr=MLM_LR, total_steps=total_steps, pct_start=0.05
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    if COMPILE_MODEL and hasattr(torch, "compile"):
        model = torch.compile(model)

    mlm_losses = []
    for epoch in range(1, MLM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with AMP_CTX:
                logits = model.forward_mlm(input_ids, attention_mask)
                loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        mlm_losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  MLM Epoch {epoch}/{MLM_EPOCHS} | loss={avg_loss:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(mlm_losses) + 1), mlm_losses, "g-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MLM Loss")
    ax.set_title("MLM Pre-training Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Save encoder + MLM weights only (exclude classification head / pool query)
    pretrained_state = {
        k: v.cpu().clone() for k, v in model.state_dict().items()
        if not k.startswith("cls_")
    }
    print(f"MLM pre-training complete. Saved {len(pretrained_state)} parameter tensors.")

    del model, optimizer, scheduler
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return pretrained_state


# Use all reports (train + test, with duplicates for more data), cleaned
all_texts_for_mlm = pd.concat(
    [pd.read_csv(TRAIN_PATH)["report"], pd.read_csv(TEST_PATH)["report"]]
).fillna("").apply(clean_text).tolist()

print(f"MLM corpus: {len(all_texts_for_mlm)} reports")
pretrained_state = pretrain_mlm(all_texts_for_mlm)


# ── 7. Training & evaluation helpers ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with AMP_CTX:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def predict_logits(model, loader) -> np.ndarray:
    """Return raw logits (n_samples, NUM_CLASSES) for ensembling."""
    model.eval()
    all_logits = []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with AMP_CTX:
            logits = model(input_ids, attention_mask)
        all_logits.append(logits.float().cpu().numpy())
    return np.concatenate(all_logits)


def predict(model, loader) -> np.ndarray:
    """Return argmax predictions."""
    return predict_logits(model, loader).argmax(axis=-1)


def plot_training_curves(train_losses, val_f1s, title="Training"):
    """Plot train loss and val macro-F1 side by side."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, "b-o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title(f"{title} — Train Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1s, "r-o", markersize=4)
    best_epoch = int(np.argmax(val_f1s)) + 1
    best_val = max(val_f1s)
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
    ax2.annotate(f"Best: {best_val:.4f} (ep {best_epoch})",
                 xy=(best_epoch, best_val), fontsize=9,
                 xytext=(best_epoch + 0.5, best_val - 0.03))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Macro-F1")
    ax2.set_title(f"{title} — Val Macro-F1")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ── 7b. Quick debug run (single train/val split, no CV) ──────────────────────

def run_debug(train_df: pd.DataFrame, val_frac=0.2):
    """Single train/val split for fast iteration. No submission."""
    from sklearn.model_selection import train_test_split

    X_text = train_df["text"].values
    y = train_df["target"].astype(int).values

    tr_idx, va_idx = train_test_split(
        np.arange(len(y)), test_size=val_frac, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\nDebug run: train={len(tr_idx)}, val={len(va_idx)}")

    train_ds = MammoDataset(X_text[tr_idx].tolist(), y[tr_idx])
    val_ds = MammoDataset(X_text[va_idx].tolist(), y[va_idx])

    if USE_BALANCED_SAMPLER:
        train_loader = make_train_loader(
            train_ds, y[tr_idx], BATCH_SIZE, sampler_seed=RANDOM_STATE
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    model = build_model()
    model.load_state_dict(pretrained_state, strict=False)
    model = model.to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()

    lr_enc = LEARNING_RATE * LR_ENCODER_MULT
    optimizer = model.configure_optimizers_discriminative(
        WEIGHT_DECAY, lr_enc, LEARNING_RATE, (0.9, 0.95), DEVICE_TYPE
    )

    if COMPILE_MODEL and hasattr(torch, "compile"):
        model = torch.compile(model)
    total_steps = len(train_loader) * EPOCHS
    max_lrs = [g["lr"] for g in optimizer.param_groups]
    scheduler = OneCycleLR(
        optimizer, max_lr=max_lrs, total_steps=total_steps, pct_start=0.1
    )

    train_losses, val_f1s = [], []
    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion)
        preds = predict(model, val_loader)
        f1 = f1_score(y[va_idx], preds, average="macro")
        train_losses.append(loss)
        val_f1s.append(f1)
        marker = " *" if f1 > best_f1 else ""
        best_f1 = max(best_f1, f1)
        print(f"  Epoch {epoch}/{EPOCHS} | loss={loss:.4f} | val macro-F1={f1:.5f}{marker}")

    print(f"\nBest val macro-F1: {best_f1:.5f}")
    print("\nClassification report:")
    preds = predict(model, val_loader)
    print(classification_report(y[va_idx], preds, digits=4))

    plot_training_curves(train_losses, val_f1s, title="Debug Run")
    return best_f1


# ── 8. StratifiedKFold CV + fold-model ensemble submission ───────────────────

def run_cv_and_submit(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_text = train_df["text"].values
    y = train_df["target"].astype(int).values
    X_test = test_df["text"].values.tolist()

    test_ds = MammoDataset(X_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = []
    oof_preds = np.full(len(train_df), -1, dtype=int)
    test_logits_sum = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float64)

    model_params = None  # print param count once

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text, y), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{N_SPLITS}  (train={len(tr_idx)}, val={len(va_idx)})")
        print(f"{'='*60}")

        train_ds = MammoDataset(X_text[tr_idx].tolist(), y[tr_idx])
        val_ds = MammoDataset(X_text[va_idx].tolist(), y[va_idx])

        if USE_BALANCED_SAMPLER:
            train_loader = make_train_loader(
                train_ds, y[tr_idx], BATCH_SIZE, sampler_seed=RANDOM_STATE + fold * 997
            )
        else:
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

        model = build_model()
        model.load_state_dict(pretrained_state, strict=False)
        model = model.to(DEVICE)
        if model_params is None:
            model_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {model_params:,}")

        criterion = nn.CrossEntropyLoss()

        lr_enc = LEARNING_RATE * LR_ENCODER_MULT
        optimizer = model.configure_optimizers_discriminative(
            WEIGHT_DECAY, lr_enc, LEARNING_RATE, (0.9, 0.95), DEVICE_TYPE
        )

        if COMPILE_MODEL and hasattr(torch, "compile"):
            model = torch.compile(model)
        total_steps = len(train_loader) * EPOCHS
        max_lrs = [g["lr"] for g in optimizer.param_groups]
        scheduler = OneCycleLR(
            optimizer, max_lr=max_lrs, total_steps=total_steps, pct_start=0.1
        )

        fold_train_losses, fold_val_f1s = [], []
        best_f1 = 0.0
        best_state = None
        no_improve = 0
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion
            )
            preds = predict(model, val_loader)
            f1 = f1_score(y[va_idx], preds, average="macro")
            fold_train_losses.append(loss)
            fold_val_f1s.append(f1)
            print(f"  Epoch {epoch}/{EPOCHS} | loss={loss:.4f} | val macro-F1={f1:.5f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                    break

        plot_training_curves(fold_train_losses, fold_val_f1s, title=f"Fold {fold}")

        # Reload best checkpoint for this fold
        model.load_state_dict(best_state)
        model.to(DEVICE)

        oof_preds[va_idx] = predict(model, val_loader)
        fold_scores.append(best_f1)
        print(f"  >> Best fold {fold} macro-F1: {best_f1:.5f}")

        test_logits_sum += predict_logits(model, test_loader)

        del model, optimizer, scheduler, criterion, best_state
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── CV results ──
    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    oof_f1 = f1_score(y, oof_preds, average="macro")

    print(f"\nCV mean: {cv_mean:.5f} | CV std: {cv_std:.5f}")
    print(f"OOF macro F1: {oof_f1:.5f}")
    print("\nClassification report (OOF):")
    print(classification_report(y, oof_preds, digits=4))

    # ── Ensemble submission ──
    test_preds = (test_logits_sum / N_SPLITS).argmax(axis=-1).astype(int)

    submission = pd.DataFrame({
        "ID": test_df["ID"],
        "target": test_preds,
    })

    assert submission["target"].isin(range(NUM_CLASSES)).all(), "Invalid BI-RADS values!"

    submission.to_csv("submission.csv", index=False)
    print(f"\nEnsemble submission saved ({N_SPLITS} folds averaged)")
    print(submission)

    return oof_f1, submission


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_cv_and_submit(train_df, test_df)
