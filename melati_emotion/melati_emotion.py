"""Melati's emotion classifier: which of seven emotions a stretch of speech carries, and when to tag it.

The model (modeling_melati_head.py, one checkpoint) is the encoder of openai/whisper-large-v3 with a head Melati
trained on its frozen features (https://github.com/Scicom-AI-Enterprise-Organization/Melati). A clip gets the most
probable emotion; a non-neutral one is kept only where it clears its threshold, chosen for 75% precision at the rate
that emotion is assumed to have in podcast speech (config `priors`). A label without a threshold is never returned.

    from melati_emotion import Emotion
    em = Emotion.from_pretrained("Scicom-intl/Melati-Emotion-Whisper-Large-V3")
    probs = em.classify([clip])          # (N, 7) over em.labels; a clip is 16 kHz mono float32, at most 30 s
    em.label(probs[0])                   # "happy", ... or "neutral"
"""
from __future__ import annotations

import numpy as np

SR = 16000


class Emotion:
    def __init__(self, path, device="cuda"):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        self.model = AutoModel.from_pretrained(str(path), trust_remote_code=True, dtype=torch.float16).to(device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(str(path))
        cfg = self.model.config
        self.labels = tuple(cfg.labels)
        self.thresholds = dict(cfg.thresholds)
        self.priors = dict(cfg.priors)
        self.device = device

    @classmethod
    def from_pretrained(cls, repo, device="cuda", revision=None):
        from huggingface_hub import snapshot_download
        return cls(snapshot_download(repo, revision=revision), device)

    def classify(self, clips, batch=32):
        """(N, len(labels)) probabilities for 16 kHz mono clips (at most 30 s; shorter than 0.1 s is padded)."""
        import torch
        clips = [np.asarray(x, np.float32)[: 30 * SR] for x in clips]
        clips = [x if len(x) >= SR // 10 else np.pad(x, (0, SR // 10 - len(x))) for x in clips]
        out = []
        with torch.inference_mode():
            for s in range(0, len(clips), batch):
                part = clips[s:s + batch]
                f = self.fe(part, sampling_rate=SR, return_tensors="pt").input_features.to(self.device)
                n = torch.tensor([max(1, min(1500, int(len(x) / SR * 50))) for x in part], device=self.device)
                out.append(self.model(f, num_frames=n).logits.softmax(-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), np.float32)

    def label(self, probs):
        """The most probable emotion, or "neutral" when it is not neutral and does not clear its threshold."""
        k = int(np.argmax(probs))
        lab = self.labels[k]
        t = self.thresholds.get(lab)
        return lab if lab == "neutral" or (t is not None and probs[k] >= t) else "neutral"
