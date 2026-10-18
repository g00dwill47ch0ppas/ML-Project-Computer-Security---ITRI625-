"""Shared preprocessing + model code for the ITRI625 phishing e-mail detector.

This file is written by the training notebook (%%writefile) and imported by both
the notebook and the FastAPI service, so training and serving can never drift apart.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_TOKEN = re.compile(r"<url>|<email>|[a-z]+(?:'[a-z]+)?|\d+|[!$%]")


# --------------------------------------------------------------------------- #
# Text preprocessing
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Lower-case and replace URLs / e-mail addresses with placeholder tokens."""
    text = str(text).lower()
    text = _URL.sub(" <url> ", text)
    text = _EMAIL.sub(" <email> ", text)
    return text


def tokenize(text: str) -> list:
    return _TOKEN.findall(clean_text(text))


def build_vocab(token_lists, max_size=30000, min_freq=2) -> dict:
    """Build the vocabulary from TRAINING tokens only (prevents leakage)."""
    counts = Counter(t for toks in token_lists for t in toks)
    itos = [PAD, UNK] + [w for w, c in counts.most_common(max_size - 2) if c >= min_freq]
    return {w: i for i, w in enumerate(itos)}


def encode_batch(texts, vocab: dict, max_len: int):
    """Convert raw texts to (padded id tensor, true-length tensor)."""
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    lengths = np.ones(len(texts), dtype=np.int64)
    for i, text in enumerate(texts):
        toks = tokenize(text)[:max_len]
        seq = [vocab.get(t, UNK_ID) for t in toks] or [UNK_ID]
        ids[i, : len(seq)] = seq
        lengths[i] = len(seq)
    return torch.from_numpy(ids), torch.from_numpy(lengths)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class PhishingBiLSTM(nn.Module):
    """Embedding -> BiLSTM -> [mean-pool ; max-pool] -> Dense(ReLU) -> logit."""

    def __init__(self, vocab_size, embed_dim=128, hidden=128, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden * 4, 64)  # mean(2h) + max(2h)
        self.act = nn.ReLU()
        self.out = nn.Linear(64, 1)

    def forward(self, ids, lengths):
        emb = self.drop(self.embedding(ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=ids.size(1))

        steps = torch.arange(ids.size(1), device=ids.device)[None, :]
        mask = (steps < lengths[:, None]).unsqueeze(-1)  # (B, T, 1)
        mean = (out * mask).sum(1) / lengths.unsqueeze(1).float()
        mx = out.masked_fill(~mask, -1e9).max(1).values

        h = self.drop(torch.cat([mean, mx], dim=1))
        h = self.drop(self.act(self.fc1(h)))
        return self.out(h).squeeze(1)  # raw logit


# --------------------------------------------------------------------------- #
# Inference wrapper (used by the API and the XAI section of the notebook)
# --------------------------------------------------------------------------- #
class PhishingClassifier:
    def __init__(self, model, vocab, config, device="cpu"):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.config = config
        self.device = device
        self.threshold = float(config.get("threshold", 0.5))

    @classmethod
    def from_dir(cls, directory="artifacts", device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        d = Path(directory)
        config = json.loads((d / "config.json").read_text())
        vocab = json.loads((d / "vocab.json").read_text())
        model = PhishingBiLSTM(len(vocab), config["embed_dim"], config["hidden"], config["dropout"])
        model.load_state_dict(torch.load(d / "model.pt", map_location=device))
        return cls(model, vocab, config, device)

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64) -> np.ndarray:
        """Return P(phishing) for each text."""
        if isinstance(texts, str):
            texts = [texts]
        probs = []
        for i in range(0, len(texts), batch_size):
            ids, lengths = encode_batch(texts[i : i + batch_size], self.vocab, self.config["max_len"])
            logits = self.model(ids.to(self.device), lengths.to(self.device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs)

    def predict(self, text: str) -> dict:
        p = float(self.predict_proba([text])[0])
        is_phish = p >= self.threshold
        return {
            "label": "phishing" if is_phish else "legitimate",
            "is_phishing": bool(is_phish),
            "probability": p,
            "threshold": self.threshold,
        }

    # -- Explainable AI (LIME) ------------------------------------------------ #
    def lime_explanation(self, text: str, num_features=10, num_samples=500):
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=["legitimate", "phishing"], random_state=42)

        def predict_fn(batch):
            p = self.predict_proba(list(batch))
            return np.column_stack([1.0 - p, p])

        return explainer.explain_instance(
            text, predict_fn, num_features=num_features, num_samples=num_samples
        )

    def explain(self, text: str, num_features=10, num_samples=500) -> list:
        """Return [{word, weight}] - positive weight pushes towards 'phishing'."""
        exp = self.lime_explanation(text, num_features, num_samples)
        return [{"word": w, "weight": float(s)} for w, s in exp.as_list()]
