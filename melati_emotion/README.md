---
license: other
license_name: melati-emotion-research
license_link: LICENSE
base_model: openai/whisper-large-v3
library_name: transformers
pipeline_tag: audio-classification
tags:
- audio-classification
- speech-emotion-recognition
- emotion
- speech-data
---

# Melati emotion classifier (Whisper-large-v3)

Which of seven emotions (neutral, happy, sad, angry, surprised, fearful, disgusted) a stretch of speech carries. Melati's speech-data pipeline
puts emotion scopes into podcast transcripts (`<happy>...</happy>`); this is the classifier behind them that can be
released, trained only on corpora whose licence allows it.

One checkpoint: the encoder of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) (float16) and a
small head trained on its frozen features (float32): every encoder layer mean-pooled over the clip's own frames, a
learned softmax mix of the layers, LayerNorm, one 256-unit hidden layer and seven outputs. Code, training and
evaluation: [Melati on GitHub](https://github.com/Scicom-AI-Enterprise-Organization/Melati).

## Use

```python
# pip install torch transformers huggingface_hub
import torch
from transformers import AutoFeatureExtractor, AutoModel

repo = "Scicom-intl/Melati-Emotion-Whisper-Large-V3"
model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.float16).cuda().eval()
fe = AutoFeatureExtractor.from_pretrained(repo)

f = fe([clip_16k], sampling_rate=16000, return_tensors="pt").input_features.cuda()
frames = torch.tensor([int(len(clip_16k) / 16000 * 50)])     # Whisper pads to 30 s: pool over the clip only
probs = model(f, num_frames=frames).logits.softmax(-1)      # over model.config.labels
```

`melati_emotion.py` (in this repository) applies the thresholds the way the pipeline does: the most probable emotion,
kept only if it is neutral or clears its threshold.

```python
from huggingface_hub import hf_hub_download
import importlib.util
spec = importlib.util.spec_from_file_location("melati_emotion", hf_hub_download(repo, "melati_emotion.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
em = m.Emotion.from_pretrained(repo)
em.label(em.classify([clip_16k])[0])        # "happy", ... or "neutral"
```

## Evaluation

On 7,791 held-out clips that none of the models below trained on (nEMO, Polish, acted; and the VoxCeleb half of
NonverbalTTS, English interviews; emotion2vec+ was fine-tuned on EmoBox, which holds neither):

<img src="https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3/resolve/main/figures/held_out.png" alt="Held-out unweighted average recall" width="760">

| held out | this model, UAR | accuracy |
|---|---|---|
| nEMO | 0.559 | 57% |
| NonverbalTTS, VoxCeleb half | 0.520 | 66% |
| both | 0.538 | 61% |

**When a label is tagged.** Acted corpora are balanced; podcasts are mostly neutral, so a threshold that is 80%
precise when a seventh of the clips are angry is far less so when one in twenty is. Each threshold is the lowest giving
75% precision once precision is re-weighted to an assumed podcast rate for that emotion; chosen on one half of the
held-out clips and measured on the other, it holds:

<img src="https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3/resolve/main/figures/recall.png" alt="Recall at 75% precision, re-weighted to podcast rates" width="589">

| emotion | threshold | assumed podcast rate | recall | precision at that rate |
|---|---|---|---|---|
| happy | 0.83 | 15% | 32% | 75% |
| sad | 0.93 | 5% | 20% | 76% |
| angry | 0.64 | 5% | 35% | 77% |
| surprised | 0.3 | 5% | 11% | 81% |
| fearful | 0.84 | 2% | 3% | 70% |
| disgusted | not tagged | 2% | n/a | n/a |

Disgust has no held-out examples, so it has no threshold and is never returned.

**Across corpora.** Each training corpus scored by the head trained on the other eight (mean UAR
0.517):

<img src="https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3/resolve/main/figures/leave_one_out.png" alt="Leave one corpus out" width="710">

## Training data

| corpus | clips | licence |
|---|---|---|
| EMNS, English, acted | 1,063 | Apache-2.0 |
| Emozionalmente, Italian, crowd-acted | 6,902 | CC BY 4.0 |
| JL-Corpus, New Zealand English, acted | 1,440 | CC0 |
| MESD, Mexican Spanish, acted | 862 | CC BY 4.0 |
| Oreau, French, acted | 502 | CC BY 4.0 |
| RESD, Russian, acted dialogues | 1,396 | MIT |
| SUBESCO, Bangla, acted | 7,000 | CC BY 4.0 |
| EmoV-DB, English, acted | 5,172 | EmoV-DB non-commercial licence |
| Expresso, English, read and improvised | 4,558 | CC BY-NC 4.0 |

Corpora are weighted equally in the loss, and classes equally within each corpus. Melati's internal pipeline also
trains on corpora that cannot be redistributed this way (CREMA-D (ODbL), RAVDESS, CaFE and PAVOQUE (CC BY-NC-SA 4.0), MELD (GPL-3.0, clipped from a TV series), NonverbalTTS's labels (CC BY-NC-SA 4.0), TurEV-DB (CC BY-NC-ND 4.0), and eNTERFACE, EmoDB, ASED, ShEMO and URDU (no licence text)); that head scores
0.557 on the same held-out clips.

## Limitations

- The training speech is mostly acted, in six languages (Bangla, English, French, Italian, Russian and Spanish), none of them Malay. On Malaysian podcasts, where Melati runs
  it, its labels are not yet measured against human judgements, and the podcast rates behind the thresholds are
  assumptions.
- Surprise and fear are found rarely at this precision; disgust never.
- One label per clip: emotion that changes inside a clip is averaged.

## Licence

Non-commercial research only; see [LICENSE](LICENSE).

## Citation

```
@misc{zolkepli2026melati,
  title  = {Melati: Multilingual Expressive speech, Labelled for Affect, Tags and Interjections},
  author = {Husein Zolkepli},
  year   = {2026},
  note   = {Scicom (MSC) Berhad}
}
```
