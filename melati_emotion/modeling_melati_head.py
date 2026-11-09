"""A Melati classifier in one checkpoint: the Whisper-large-v3 encoder and the head trained on its frozen features.

Both of Melati's released classifiers (non-verbal sounds, emotion) are this model; the head is what Melati's pipeline
runs (melati/ser.py): every encoder layer (and the embeddings) mean-pooled over the clip's own frames, a learned
softmax mix of the layers, LayerNorm, one hidden layer with GELU, and one output per label. Whisper pads every input to 30 s, so pass `num_frames` (50 per second of the clip) or the padding is averaged
in. The pooled features were stored as float16 when the head was trained, so they are rounded the same way here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, WhisperConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder

try:
    from .configuration_melati_head import MelatiHeadConfig
except ImportError:                                    # imported as a plain module, outside a package
    from configuration_melati_head import MelatiHeadConfig


class MelatiHeadModel(PreTrainedModel):
    config_class = MelatiHeadConfig
    main_input_name = "input_features"
    # the head runs in float32, as it was trained; its names must not match any encoder module (Whisper's layers
    # have their own fc1 / fc2, which would then stay float32 inside a float16 encoder)
    _keep_in_fp32_modules = ["layer_mix", "head_norm", "head_fc1", "head_out"]

    def __init__(self, config):
        super().__init__(config)
        enc = WhisperConfig(**config.encoder)
        self.encoder = WhisperEncoder(enc)
        self.layer_mix = nn.Parameter(torch.zeros(enc.encoder_layers + 1))
        self.head_norm = nn.LayerNorm(enc.d_model)
        self.head_fc1 = nn.Linear(enc.d_model, config.hidden)
        self.head_out = nn.Linear(config.hidden, len(config.labels))
        self.post_init()

    def _init_weights(self, module):
        pass                                       # every weight comes from the checkpoint

    def pooled(self, input_features, num_frames=None):
        """(B, layers, dim): each encoder layer averaged over the first num_frames frames of its clip."""
        enc_dtype = self.encoder.conv1.weight.dtype
        h = torch.stack(self.encoder(input_features.to(enc_dtype), output_hidden_states=True).hidden_states, 1)
        t = h.shape[2]
        if num_frames is None:
            num_frames = torch.full((h.shape[0],), t)
        n = torch.as_tensor(num_frames, device=h.device).clamp(1, t)
        mask = (torch.arange(t, device=h.device)[None, :] < n[:, None]).float()          # B, T
        x = (h.float() * mask[:, None, :, None]).sum(2) / n[:, None, None].float()
        return x.half().float()

    def forward(self, input_features, num_frames=None, labels=None, **kwargs):
        x = self.pooled(input_features, num_frames)
        z = (self.layer_mix.float().softmax(0)[None, :, None] * x).sum(1)
        z = F.layer_norm(z, (z.shape[-1],), self.head_norm.weight.float(), self.head_norm.bias.float(),
                         self.head_norm.eps)
        z = F.gelu(F.linear(z, self.head_fc1.weight.float(), self.head_fc1.bias.float()))
        logits = F.linear(z, self.head_out.weight.float(), self.head_out.bias.float())
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)
