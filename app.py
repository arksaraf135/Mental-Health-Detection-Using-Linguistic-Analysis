import os
import re

from flask import Flask, render_template, request

try:
    import torch  # type: ignore
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer  # type: ignore[reportMissingImports]
except ImportError as e:
    raise RuntimeError(
        "Missing dependencies. Install: torch transformers\n"
        "Example: pip install torch transformers"
    ) from e

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ======================== CONFIG ========================
SAVED_MODEL_DIR = "transformer_depression_model"
SAVED_FULL_MODEL_FILE = "safm_net.pt"

HIGH_THRESHOLD = 0.65
MEDIUM_THRESHOLD = 0.35

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

sentiment_analyzer = SentimentIntensityAnalyzer()

# Hybrid stability rules: this layer reduces medium-band volatility that is common
# when training data is binary but user-facing output has 3 risk tiers.
CRISIS_PHRASES = {
    "i dont want to be alive",
    "i don't want to be alive",
    "want to end my life",
    "end my life",
    "kill myself",
    "hurt myself",
    "i might act on it",
    "i have the means",
    "suicide",
}

DISTRESS_PHRASES = {
    "drained",
    "unmotivated",
    "empty",
    "overwhelmed",
    "disconnected",
    "hopeless",
    "struggling",
    "hard time concentrating",
    "hard time",
    "numb",
    "exhausted",
}

FUNCTIONING_PHRASES = {
    "still trying",
    "still go to class",
    "still go to work",
    "still work",
    "still able",
    "get through my day",
    "keep up with my responsibilities",
    "taking care of myself",
    "trying to function",
    "i can still",
}

PROTECTIVE_PHRASES = {
    "hopeful",
    "grateful",
    "calm",
    "feeling better",
    "doing okay",
}

DISTRESS_TOKENS = {
    "drained",
    "empty",
    "hopeless",
    "overwhelmed",
    "unmotivated",
    "struggling",
    "numb",
    "anxious",
    "worthless",
    "lonely",
    "disconnected",
}

FUNCTIONING_TOKENS = {
    "still",
    "trying",
    "function",
    "working",
    "work",
    "class",
    "school",
    "responsibilities",
    "routine",
    "managing",
    "cope",
}


# ======================== LINGUISTIC FEATURE EXTRACTOR ========================

def extract_linguistic_features(text: str) -> np.ndarray:
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


# ======================== SAFM-NET MODEL (must match training architecture) ========================

class CrossAttentionFusion(nn.Module):
    def __init__(self, token_dim: int, ling_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.ling_proj = nn.Linear(ling_dim, token_dim)
        self.attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, token_embeddings: torch.Tensor, ling_features: torch.Tensor):
        query = self.ling_proj(ling_features).unsqueeze(1)
        attn_out, _ = self.attn(query, token_embeddings, token_embeddings)
        return self.norm(attn_out.squeeze(1))


class GatedMerge(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, cls_emb: torch.Tensor, attn_emb: torch.Tensor):
        g = torch.sigmoid(self.gate(torch.cat([cls_emb, attn_emb], dim=-1)))
        return g * cls_emb + (1 - g) * attn_emb


class SAFMNet(nn.Module):
    def __init__(self, transformer_name: str, num_ling_features: int,
                 fusion_heads: int, fusion_dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(transformer_name)
        hidden = self.encoder.config.hidden_size

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
        token_embs = enc_out.last_hidden_state
        cls_emb = token_embs[:, 0, :]

        attn_emb = self.cross_attn(token_embs, ling_features)
        fused = self.gate(cls_emb, attn_emb)
        fused = self.dropout(fused)

        binary_logits = self.binary_head(fused)
        ordinal_logits = self.ordinal_head(fused)
        return binary_logits, ordinal_logits, fused


# ======================== RULE-BASED STABILITY LAYER ========================

def _normalize_text(text: str) -> str:
    normalized = (text or "").lower()
    # Normalize smart punctuation variants to improve phrase matching.
    normalized = (
        normalized.replace("’", "'")
        .replace("`", "'")
        .replace("“", "\"")
        .replace("”", "\"")
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _count_matches(text: str, phrases: set[str]) -> int:
    return sum(1 for p in phrases if p in text)

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text)


def stabilize_probability(cleaned_text: str, probability: float) -> float:
    """
    Stabilize final probabilities for clinically intuitive 3-tier behavior:
      - explicit crisis language -> high-risk floor
      - distress + functioning coexistence -> medium band
      - clearly protective language without distress -> low cap
    """
    normalized = _normalize_text(cleaned_text)
    tokens = _tokenize(normalized)
    token_set = set(tokens)

    crisis_hits = _count_matches(normalized, CRISIS_PHRASES)
    distress_hits = _count_matches(normalized, DISTRESS_PHRASES)
    functioning_hits = _count_matches(normalized, FUNCTIONING_PHRASES)
    protective_hits = _count_matches(normalized, PROTECTIVE_PHRASES)

    distress_token_hits = len(token_set.intersection(DISTRESS_TOKENS))
    functioning_token_hits = len(token_set.intersection(FUNCTIONING_TOKENS))

    # Pattern: "distress clause, but still coping clause"
    coping_bridge = bool(
        re.search(
            r"\bbut\b.*\b(still|trying|can still|keep|manag|cope)\b",
            normalized,
        )
    )

    # Extra sentiment signal for stabilization only.
    compound = sentiment_analyzer.polarity_scores(normalized)["compound"]

    distress_score = distress_hits + distress_token_hits + (1 if compound < -0.3 else 0)
    functioning_score = functioning_hits + functioning_token_hits + (1 if coping_bridge else 0)

    # Never allow explicit crisis language to stay low/medium.
    if crisis_hits > 0:
        return max(probability, 0.80)

    # Distress + intact functioning should map to Medium more consistently.
    if distress_score >= 2 and functioning_score >= 1:
        return min(max(probability, 0.43), 0.61)

    # Explicit "but still coping" pattern is strongly medium-like.
    if coping_bridge and distress_score >= 1:
        return min(max(probability, 0.44), 0.60)

    # Strong distress with no functioning cues should not collapse to very low.
    if distress_score >= 3 and functioning_score == 0:
        return max(probability, 0.55)

    # Mild distress should not fall unrealistically low.
    if distress_score >= 2 and probability < 0.30:
        return 0.36

    # Clearly protective/no-distress phrasing should not drift high.
    if protective_hits >= 2 and distress_score == 0:
        return min(probability, 0.28)

    return probability


# ======================== LOAD ========================

app = Flask(__name__)


def load_artifacts():
    ckpt_path = os.path.join(SAVED_MODEL_DIR, SAVED_FULL_MODEL_FILE)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["model_config"]

    tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_DIR)
    model = SAFMNet(
        transformer_name=cfg["transformer_name"],
        num_ling_features=cfg["num_ling_features"],
        fusion_heads=cfg["fusion_heads"],
        fusion_dropout=cfg["fusion_dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    max_length = int(checkpoint.get("max_length", 128))
    high_t = float(checkpoint.get("high_threshold", HIGH_THRESHOLD))
    medium_t = float(checkpoint.get("medium_threshold", MEDIUM_THRESHOLD))
    return tokenizer, model, device, max_length, high_t, medium_t


tokenizer, safm_model, DEVICE, MAX_LENGTH, HIGH_T, MEDIUM_T = load_artifacts()


# ======================== INFERENCE ========================

def predict_risk(text: str):
    cleaned = (text or "").strip()
    if not cleaned:
        return "Low Risk", 0.0

    inputs = tokenizer(cleaned, truncation=True, padding="max_length",
                       max_length=MAX_LENGTH, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    ling = torch.tensor(extract_linguistic_features(cleaned), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        binary_logits, ordinal_logits, _ = safm_model(
            inputs["input_ids"], inputs["attention_mask"], ling
        )

    # Binary probability from classification head
    binary_prob = F.softmax(binary_logits, dim=1)[0, 1].item()

    # Ordinal cumulative probabilities: P(>=Med) and P(>=High)
    ord_probs = torch.sigmoid(ordinal_logits[0])
    p_ge_med = ord_probs[0].item()
    p_ge_high = min(ord_probs[1].item(), p_ge_med)  # monotonicity constraint

    # Proper tier probabilities from ordinal head
    p_high = p_ge_high
    p_med = p_ge_med - p_ge_high
    p_low = 1.0 - p_ge_med

    # Expected risk score: Low=0, Medium=0.5, High=1.0
    ordinal_risk = 0.0 * p_low + 0.5 * p_med + 1.0 * p_high

    # Blend: ordinal drives tier placement, binary provides confidence
    probability = 0.65 * ordinal_risk + 0.35 * binary_prob
    probability = stabilize_probability(cleaned, probability)

    if probability >= HIGH_T:
        risk_label = "High Risk"
    elif probability >= MEDIUM_T:
        risk_label = "Medium Risk"
    else:
        risk_label = "Low Risk"

    return risk_label, probability


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    user_text = ""
    if request.method == "POST":
        user_text = request.form["text"]
        result, probability = predict_risk(user_text)
    return render_template("index.html", result=result, probability=probability, user_text=user_text)


if __name__ == "__main__":
    app.run(debug=True)
