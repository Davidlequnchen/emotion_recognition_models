"""Configuration of a Melati head on the Whisper-large-v3 encoder (the non-verbal and the emotion classifiers)."""
from transformers import PretrainedConfig


class MelatiHeadConfig(PretrainedConfig):
    model_type = "melati_head"

    def __init__(self, encoder=None, labels=None, hidden=256, thresholds=None, candidate_rule=None, priors=None,
                 task=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder or {}                  # openai/whisper-large-v3's WhisperConfig, as a dict
        self.labels = list(labels or [])              # the head's outputs, in order
        self.hidden = hidden                          # the head's hidden layer
        self.thresholds = dict(thresholds or {})      # per label, chosen for a stated precision
        self.candidate_rule = dict(candidate_rule or {})   # non-verbal: how a sound in a word gap becomes a candidate
        self.priors = dict(priors or {})              # emotion: the podcast rates its precision was re-weighted to
        self.task = task                              # "nonverbal" or "emotion"
