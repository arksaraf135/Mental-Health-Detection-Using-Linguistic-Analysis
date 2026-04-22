"""
SAFM-Net: Sentiment-Augmented Fusion Model with Multi-Granular Attention
        for Mental Health Risk Stratification.

Architecture:
  Text -> DistilBERT encoder (token embeddings + [CLS])
       -> Parallel linguistic feature extraction (VADER + lexical markers)
       -> Cross-attention fusion (linguistic features attend over tokens)
       -> Gated merge with [CLS]
       -> Dual head: binary depression + ordinal risk (Low/Medium/High)
       -> Composite loss: Focal + Ordinal CE + Brier
"""

import os
import re
import math

import numpy as np
import pandas as pd

try:
    import torch  # type: ignore
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import (  # type: ignore[reportMissingImports]
        AutoModel,
        AutoTokenizer,
    )
except ImportError as e:
    raise RuntimeError(
        "Missing dependencies. Install: torch transformers\n"
        "Example: pip install torch transformers"
    ) from e

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, brier_score_loss
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ======================== CONFIG ========================
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
SAVED_MODEL_DIR = "transformer_depression_model"
SAVED_FULL_MODEL_FILE = "safm_net.pt"

EPOCHS = 4
ENCODER_LR = 2e-5
HEAD_LR = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
CV_SEED = 42

LOSS_ALPHA = 1.0   # focal loss weight
LOSS_BETA = 0.8    # ordinal loss weight
LOSS_GAMMA = 0.3   # brier loss weight
FOCAL_GAMMA = 2.0

NUM_LINGUISTIC_FEATURES = 8
FUSION_HEADS = 4
FUSION_DROPOUT = 0.15

HIGH_THRESHOLD = 0.65
MEDIUM_THRESHOLD = 0.35

WELLBEING_HARD_NEGATIVES = [
    "I have been feeling calm and grounded this week and I am hopeful about what comes next.",
    "Today was not perfect, but I handled my responsibilities and I still feel okay overall.",
    "I feel grateful for small things lately and I am trying to stay consistent with my routine.",
    "Some moments are stressful, yet I can manage them and keep moving forward.",
    "I am tired sometimes, but I still find motivation and purpose in my day.",
    "I feel at peace more often now and I am excited to keep improving.",
    "I had a rough day but I reached out to a friend and now I feel better.",
    "I am learning to cope with pressure and I can still enjoy parts of my day.",
    "I feel stable right now and I am making plans I genuinely look forward to.",
    "I am not always energetic, but I feel emotionally safe and supported.",
    "I can feel stress at times, but I am functioning and recovering well.",
    "I feel hopeful, connected, and capable of handling what is ahead.",
    "I am doing okay, sleeping better, and I can focus on my goals again.",
    "I feel emotionally balanced today and I am grateful for my progress.",
    "I can acknowledge difficult feelings without feeling overwhelmed by them.",
]

ABSOLUTIST_WORDS = {
    "always", "never", "nothing", "everything", "completely", "totally",
    "absolutely", "entirely", "forever", "nobody", "everyone", "impossible",
    "worthless", "hopeless", "pointless", "useless",
}
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nor",
    "nowhere", "cannot", "can't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "shouldn't", "couldn't", "isn't", "aren't",
    "wasn't", "weren't", "haven't", "hasn't", "hadn't",
}
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}


# ======================== LINGUISTIC FEATURE EXTRACTOR ========================
sentiment_analyzer = SentimentIntensityAnalyzer()


def extract_linguistic_features(text: str) -> np.ndarray:
    """Return K=8 linguistic features for one text sample."""
    scores = sentiment_analyzer.polarity_scores(text)
    words = re.findall(r"[a-z']+", text.lower())
    n = max(len(words), 1)

    pronoun_ratio = sum(1 for w in words if w in FIRST_PERSON) / n
    absolutist_ratio = sum(1 for w in words if w in ABSOLUTIST_WORDS) / n
    negation_ratio = sum(1 for w in words if w in NEGATION_WORDS) / n
    exclaim_ratio = text.count("!") / max(len(text), 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0

    return np.array([
        scores["compound"],
        scores["pos"],
        scores["neg"],
        pronoun_ratio,
        absolutist_ratio,
        negation_ratio,
        exclaim_ratio,
        avg_word_len,
    ], dtype=np.float32)


def extract_linguistic_batch(texts: list[str]) -> np.ndarray:
    return np.stack([extract_linguistic_features(t) for t in texts])


# ======================== SAFM-NET MODEL ========================

class CrossAttentionFusion(nn.Module):
    """Linguistic features (query) attend over transformer token embeddings (key/value)."""
    def __init__(self, token_dim: int, ling_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.ling_proj = nn.Linear(ling_dim, token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, token_embeddings: torch.Tensor, ling_features: torch.Tensor):
        query = self.ling_proj(ling_features).unsqueeze(1)  # [B, 1, D]
        attn_out, _ = self.attn(query, token_embeddings, token_embeddings)
        return self.norm(attn_out.squeeze(1))  # [B, D]


class GatedMerge(nn.Module):
    """Gated combination: z = g * cls + (1 - g) * attn_out"""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, cls_emb: torch.Tensor, attn_emb: torch.Tensor):
        g = torch.sigmoid(self.gate(torch.cat([cls_emb, attn_emb], dim=-1)))
        return g * cls_emb + (1 - g) * attn_emb


class SAFMNet(nn.Module):
    """
    Sentiment-Augmented Fusion Model with Multi-Granular Attention.
    Dual output: binary depression head + ordinal risk head.
    """
    def __init__(self, transformer_name: str, num_ling_features: int,
                 fusion_heads: int, fusion_dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(transformer_name)
        hidden = self.encoder.config.hidden_size  # 768 for distilbert

        self.cross_attn = CrossAttentionFusion(hidden, num_ling_features,
                                               fusion_heads, fusion_dropout)
        self.gate = GatedMerge(hidden)
        self.dropout = nn.Dropout(0.2)

        self.binary_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 2),
        )

        self.ordinal_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, input_ids, attention_mask, ling_features):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = enc_out.last_hidden_state       # [B, T, 768]
        cls_emb = token_embs[:, 0, :]                # [B, 768]

        attn_emb = self.cross_attn(token_embs, ling_features)
        fused = self.gate(cls_emb, attn_emb)
        fused = self.dropout(fused)

        binary_logits = self.binary_head(fused)       # [B, 2]
        ordinal_logits = self.ordinal_head(fused)     # [B, 2]  (cumulative)

        return binary_logits, ordinal_logits, fused

    def encoder_parameters(self):
        return self.encoder.parameters()

    def head_parameters(self):
        head_modules = [self.cross_attn, self.gate, self.dropout,
                        self.binary_head, self.ordinal_head]
        for m in head_modules:
            yield from m.parameters()


# ======================== LOSS FUNCTIONS ========================

def focal_loss(logits, targets, gamma=FOCAL_GAMMA):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def ordinal_bce(ordinal_logits, ordinal_targets):
    return F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets.float())


def brier_loss(logits, targets):
    probs = F.softmax(logits, dim=1)[:, 1]
    return ((probs - targets.float()) ** 2).mean()


def composite_loss(binary_logits, ordinal_logits, binary_targets, ordinal_targets,
                   alpha=LOSS_ALPHA, beta=LOSS_BETA, gamma=LOSS_GAMMA):
    l_focal = focal_loss(binary_logits, binary_targets)
    l_ordinal = ordinal_bce(ordinal_logits, ordinal_targets)
    l_brier = brier_loss(binary_logits, binary_targets)
    return alpha * l_focal + beta * l_ordinal + gamma * l_brier


# ======================== DATASET ========================

class SAFMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, ling_features, binary_labels, ordinal_labels):
        self.encodings = encodings
        self.ling_features = torch.tensor(ling_features, dtype=torch.float32)
        self.binary_labels = torch.tensor(binary_labels, dtype=torch.long)
        self.ordinal_labels = torch.tensor(ordinal_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.binary_labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["ling_features"] = self.ling_features[idx]
        item["binary_labels"] = self.binary_labels[idx]
        item["ordinal_labels"] = self.ordinal_labels[idx]
        return item


def assign_ordinal_labels(binary_labels: list[int], ling_features: np.ndarray) -> np.ndarray:
    """
    Assign ordinal cumulative targets [Y>=Med, Y>=High] using a data-driven
    severity threshold so that Medium and High are roughly balanced within class 1.

    Severity = weighted combination of negative sentiment, absolutist language,
    negation density, and self-focused pronouns.
    """
    compound = ling_features[:, 0]
    neg_score = ling_features[:, 2]
    absolutist = ling_features[:, 4]
    negation = ling_features[:, 5]
    pronoun = ling_features[:, 3]

    severity = (np.maximum(0, -compound) * 1.0
                + neg_score * 1.5
                + absolutist * 4.0
                + negation * 1.5
                + pronoun * 1.0)

    class1_mask = np.array(binary_labels) == 1
    class1_severity = severity[class1_mask]

    # Use median so ~50% of class 1 becomes Medium, ~50% becomes High
    threshold = float(np.median(class1_severity)) if len(class1_severity) > 0 else 0.5

    ordinal = np.zeros((len(binary_labels), 2), dtype=np.float32)
    for i, lab in enumerate(binary_labels):
        if lab == 1:
            ordinal[i, 0] = 1.0                                    # Y >= Medium
            ordinal[i, 1] = 1.0 if severity[i] >= threshold else 0.0  # Y >= High
        # label 0 stays [0, 0] (Low)

    return ordinal, threshold


# ======================== WARMUP + COSINE SCHEDULER ========================

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        super().__init__(optimizer, lr_lambda)


# ======================== TRAINING LOOP ========================

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch_num):
    model.train()
    total_loss = 0.0
    log_interval = max(1, len(loader) // 5)

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ling = batch["ling_features"].to(device)
        b_labels = batch["binary_labels"].to(device)
        o_labels = batch["ordinal_labels"].to(device)

        binary_logits, ordinal_logits, _ = model(input_ids, attention_mask, ling)
        loss = composite_loss(binary_logits, ordinal_logits, b_labels, o_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            lr_enc = optimizer.param_groups[0]["lr"]
            lr_head = optimizer.param_groups[1]["lr"]
            print(f"  [Epoch {epoch_num} step {step+1}/{len(loader)}] "
                  f"loss={loss.item():.4f}  lr_enc={lr_enc:.2e}  lr_head={lr_head:.2e}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_binary_logits, all_ordinal_logits = [], []
    all_b_labels = []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ling = batch["ling_features"].to(device)
        b_labels = batch["binary_labels"].to(device)
        o_labels = batch["ordinal_labels"].to(device)

        binary_logits, ordinal_logits, _ = model(input_ids, attention_mask, ling)
        loss = composite_loss(binary_logits, ordinal_logits, b_labels, o_labels)
        total_loss += loss.item()

        all_binary_logits.append(binary_logits.cpu())
        all_ordinal_logits.append(ordinal_logits.cpu())
        all_b_labels.append(b_labels.cpu())

    all_binary_logits = torch.cat(all_binary_logits)
    all_ordinal_logits = torch.cat(all_ordinal_logits)
    all_b_labels = torch.cat(all_b_labels)

    binary_probs = F.softmax(all_binary_logits, dim=1)[:, 1].numpy()
    ord_raw = torch.sigmoid(all_ordinal_logits).numpy()

    # Enforce monotonicity: P(>=High) <= P(>=Med)
    p_ge_med = ord_raw[:, 0]
    p_ge_high = np.minimum(ord_raw[:, 1], p_ge_med)

    # Proper tier probabilities
    p_high = p_ge_high
    p_med = p_ge_med - p_ge_high
    p_low = 1.0 - p_ge_med

    # Ordinal expected risk score: Low=0, Medium=0.5, High=1.0
    ordinal_risk = 0.0 * p_low + 0.5 * p_med + 1.0 * p_high

    # Blend: ordinal drives tier placement, binary provides confidence
    ensemble_probs = 0.65 * ordinal_risk + 0.35 * binary_probs

    preds = (ensemble_probs >= 0.5).astype(int)
    labels_np = all_b_labels.numpy()

    acc = accuracy_score(labels_np, preds)
    brier = brier_score_loss(labels_np, np.clip(ensemble_probs, 0, 1))
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, brier, ensemble_probs, labels_np


# ======================== MAIN ========================

if __name__ == "__main__":

    # ---------- LOAD & CLEAN DATA ----------
    df = pd.read_csv("dataset.csv")
    print("Original dataset size:", len(df))
    print("Original label distribution:")
    print(df["is_depression"].value_counts())

    hard_neg_df = pd.DataFrame(
        {"clean_text": WELLBEING_HARD_NEGATIVES,
         "is_depression": [0] * len(WELLBEING_HARD_NEGATIVES)}
    )
    df = pd.concat([df, hard_neg_df], ignore_index=True)
    print(f"Added hard negatives for class 0: {len(WELLBEING_HARD_NEGATIVES)}")

    df = df.dropna(subset=["clean_text", "is_depression"])
    df["clean_text"] = df["clean_text"].astype(str)
    df = df[df["clean_text"].str.strip() != ""]
    df["token_count"] = df["clean_text"].str.split().str.len()
    df = df[df["token_count"] >= 3]
    df = df[df["token_count"] <= 200]
    df = df.drop_duplicates(subset=["clean_text", "is_depression"])

    print("\nAfter cleaning size:", len(df))
    print("After cleaning label distribution:")
    print(df["is_depression"].value_counts())

    # Balance
    label_counts = df["is_depression"].value_counts()
    min_count = label_counts.min()
    balanced_frames = []
    for label, _ in label_counts.items():
        label_df = df[df["is_depression"] == label]
        if label == 0:
            hard = label_df[label_df["clean_text"].isin(WELLBEING_HARD_NEGATIVES)]
            remaining = max(0, min_count - len(hard))
            pool = label_df[~label_df.index.isin(hard.index)]
            rest = pool.sample(remaining, random_state=CV_SEED) if remaining > 0 else pool.iloc[0:0]
            balanced_frames.append(pd.concat([hard, rest], ignore_index=True))
        else:
            balanced_frames.append(label_df.sample(min_count, random_state=CV_SEED))
    df = pd.concat(balanced_frames).sample(frac=1.0, random_state=CV_SEED).reset_index(drop=True)

    print("\nAfter balancing size:", len(df))
    print("After balancing label distribution:")
    print(df["is_depression"].value_counts())

    texts = df["clean_text"].tolist()
    binary_labels = df["is_depression"].astype(int).tolist()

    # ---------- LINGUISTIC FEATURES ----------
    print("\nExtracting linguistic features...")
    ling_features_all = extract_linguistic_batch(texts)

    # Ordinal labels: data-driven severity split for balanced Medium/High
    ordinal_labels, severity_threshold = assign_ordinal_labels(binary_labels, ling_features_all)
    print(f"Severity threshold (median of class-1): {severity_threshold:.4f}")

    # Verify ordinal distribution
    ol_sums = ordinal_labels.sum(axis=0)
    n_low = int(np.sum((ordinal_labels[:, 0] == 0) & (ordinal_labels[:, 1] == 0)))
    n_med = int(np.sum((ordinal_labels[:, 0] == 1) & (ordinal_labels[:, 1] == 0)))
    n_high = int(np.sum((ordinal_labels[:, 0] == 1) & (ordinal_labels[:, 1] == 1)))
    print(f"\nOrdinal label distribution: Low={n_low}, Medium={n_med}, High={n_high}")

    # ---------- SPLIT ----------
    idx_all = np.arange(len(texts))
    idx_train, idx_temp, _, _ = train_test_split(
        idx_all, binary_labels, test_size=0.30, random_state=CV_SEED, stratify=binary_labels,
    )
    temp_labels = [binary_labels[i] for i in idx_temp]
    idx_cal, idx_test, _, _ = train_test_split(
        idx_temp, temp_labels, test_size=0.50, random_state=CV_SEED, stratify=temp_labels,
    )

    texts_train = [texts[i] for i in idx_train]
    texts_test = [texts[i] for i in idx_test]

    bl_train = [binary_labels[i] for i in idx_train]
    bl_test = [binary_labels[i] for i in idx_test]
    ol_train = ordinal_labels[idx_train]
    ol_test = ordinal_labels[idx_test]
    lf_train = ling_features_all[idx_train]
    lf_test = ling_features_all[idx_test]

    # ---------- TOKENIZE ----------
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_enc = tokenizer(texts_train, truncation=True, padding="max_length",
                          max_length=MAX_LENGTH, return_tensors="pt")
    test_enc = tokenizer(texts_test, truncation=True, padding="max_length",
                         max_length=MAX_LENGTH, return_tensors="pt")

    train_ds = SAFMDataset(train_enc, lf_train, bl_train, ol_train)
    test_ds = SAFMDataset(test_enc, lf_test, bl_test, ol_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE)

    # ---------- MODEL ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SAFMNet(MODEL_NAME, NUM_LINGUISTIC_FEATURES, FUSION_HEADS, FUSION_DROPOUT)
    model.to(device)

    # Differential learning rates: slow for pre-trained encoder, fast for custom heads
    optimizer = torch.optim.AdamW([
        {"params": model.encoder_parameters(), "lr": ENCODER_LR},
        {"params": model.head_parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

    # ---------- TRAIN ----------
    print(f"\nTraining SAFM-Net for {EPOCHS} epochs...")
    print(f"Encoder LR: {ENCODER_LR}, Head LR: {HEAD_LR}")
    best_brier = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss, val_acc, val_brier, _, _ = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_brier={val_brier:.5f}")

        if val_brier < best_brier:
            best_brier = val_brier
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> New best model (brier={best_brier:.5f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- FINAL EVALUATION ----------
    _, final_acc, final_brier, test_probs, test_labels = evaluate(model, test_loader, device)
    test_preds = (test_probs >= 0.5).astype(int)

    print("\n=== SAFM-Net Final Results ===")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Brier Score: {final_brier:.5f}")
    print("\nClassification Report:\n", classification_report(test_labels, test_preds))
    print("Confusion Matrix:\n", confusion_matrix(test_labels, test_preds))

    # ---------- SAVE ----------
    tokenizer.save_pretrained(SAVED_MODEL_DIR)

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "transformer_name": MODEL_NAME,
            "num_ling_features": NUM_LINGUISTIC_FEATURES,
            "fusion_heads": FUSION_HEADS,
            "fusion_dropout": FUSION_DROPOUT,
        },
        "max_length": MAX_LENGTH,
        "high_threshold": HIGH_THRESHOLD,
        "medium_threshold": MEDIUM_THRESHOLD,
    }, os.path.join(SAVED_MODEL_DIR, SAVED_FULL_MODEL_FILE))

    print(f"\nSaved SAFM-Net to '{SAVED_MODEL_DIR}/{SAVED_FULL_MODEL_FILE}'")
    print(f"Saved tokenizer to '{SAVED_MODEL_DIR}/'")
    print("Done.")
