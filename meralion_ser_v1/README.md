---
library_name: transformers
license: other
license_name: meralion-public-license
license_link: https://huggingface.co/MERaLiON/MERaLiON-SER-v1/resolve/main/MERaLiON-Public-Licence-SER-V1.pdf
extra_gated_fields:
  First Name: text
  Last Name: text
  Country: country
  Affiliation: text
  Job title:
    type: select
    options:
    - Student
    - Research Graduate
    - AI developer/Researcher
    - Other
  I consent to being contacted by the MERaLiON team for feedback or follow-up regarding my experience using the model: checkbox
extra_gated_description: >-
  By downloading this model, you acknowledge that you have read and agree to be bound by the Terms and Conditions set out in this document [MERaLiON Public License](https://huggingface.co/datasets/MERaLiON/MERaLiON_Public_Licence/blob/main/MERaLiON-Public-Licence-SER-V1.pdf).
  The information you provide will be collected, stored, processed, and shared in accordance with the [A*STAR Privacy Policy](https://www.a-star.edu.sg/privacy-statement).
extra_gated_button_content: Submit
tags:
- meralion-ser-v1
- speech
- best-rq
- meralion
- meralion-2
language:
- en
- zh
- ms
- ta
- id
- th
- vi
---

# MERaLiON-SER-v1: Multilingual Speech Emotion Model

## 🎤 Live Demo

You can experience **speech emotion recognition** by selecting Gender/Speech recognition tab
via our interactive Hugging Face Space:

👉 **[MERaLiON-SER Demo](https://meralion.org/demo/)**

Upload an audio clip or record your voice to visualize categorical emotions  
and dimensional affect trajectories in real time.

---

## 📘 Model Summary

**MERaLiON-SER-v1** is a multilingual **speech emotion recognition (SER)** model jointly predicting  
1. **Categorical emotions** – 7 discrete classes *(Neutral, Happy, Sad, Angry, Surprised, Fearful, Disgusted)*, and  
2. **Dimensional affect values** – continuous *(Valence, Arousal, Dominance)*.   
Valence (0 negative, 1 positive),
Arousal (0 calm, 1 active),
Dominance (0 weak, 1 strong)

The design achieves parameter-efficient adaptation for multilingual, paralinguistic affect modeling using just **309 M** parameters.

**Language(s):** English (Global & Singapore), Chinese, Malay, Tamil, Limited support for Thai, Indonesian, and Vietnamese.

More details on model architecture, training and evaluation is available here: [Technical report](http://arxiv.org/abs/2511.04914)

**License:** [MERaLiON Public License](https://huggingface.co/MERaLiON/MERaLiON-SER-v1/resolve/main/MERaLiON-Public-Licence-SER-V1.pdf)
---

## 🎯 Supported Outputs

| Head | Type | Output | Description |
|------|------|---------|-------------|
| **Categorical** | Softmax | `logits → (7)` | Neutral, Happy, Sad, Angry, Surprised, Fearful, Disgusted |
| **Dimensional** | Sigmoid | `dims → (3)` | Valence [V], Arousal [A], Dominance [D] in [0, 1] |

---

## 🧠 Architecture Overview
- **Modality:** Speech-only model
- **Backbone:** Whisper-Medium encoder (frozen) for multilingual acoustic features with LoRA adaptation. The Whisper decoder is frozen and unused.
- **Downstream:** Attention-based pooling + modified ECAPA-TDNN capturing temporal & speaker-invariant cues.  
- **Dual-head outputs:**  
  - Categorical (Softmax) → discrete emotion classes.  
  - Dimensional (Sigmoid) → continuous VAD estimation.  
- **Parameter-efficient fine-tuning:** LoRA adapters integrated into Q/K/V attention projections.  
- **Objective:** Weighted Cross-Entropy + Concordance Correlation Coefficient (CCC) loss.  
- **Pooling:** Attention pooling for combining short-term and long-term cues.  
- **Augmentations:** MixUp, speed perturbation, and additive noise for robustness.
- **Model parameters size:** 309 M

This framework balances **computational efficiency**, **cross-lingual transferability**, and **robust emotion generalization**.

---

## 📊 Performance Overview
In speech emotion recognition (SER), class imbalance is a persistent challenge, as certain emotions such as neutral or happy typically dominate spontaneous datasets, while others like fear or disgust occur infrequently. Conventional performance metrics such as weighted accuracy or overall accuracy tend to be dominated by these majority classes, often masking a model’s poor discrimination of minority emotions. To provide a more balanced evaluation, researchers in the affective computing community commonly employ **Unweighted Average Recall (UAR)** also know as **Balanced Accuracy** as the principal metric.

UAR is defined as the mean of per-class recall values, thereby assigning equal importance to each emotional category regardless of its occurrence frequency. This ensures that improvements in recognizing underrepresented emotions contribute equally to the final score. Unlike weighted accuracy—which aggregates correct predictions proportional to class distribution—UAR offers a class-independent assessment that more accurately reflects a model’s generalization capability across diverse emotional states. Consequently, UAR has become the de facto standard for benchmarking emotion recognition systems.

We have evaluated MERaLiON-SER-v1 performance using English and Singapore languages: English (Singlish), Chinese, Malay, and Tamil.
The evaluation dataset for Singapore languages containes fine-grained labels at every two seconds and merged nearby segments to create a course level segments with maximum 15 second duration.

We have also added performance of primary 4 classes in emotion literature: Neutral, Angry, Sad, and Happy.
 
### Singapore Language Performance
<img src="https://huggingface.co/MERaLiON/MERaLiON-SER-v1/resolve/main/SG_UAR_4plots_8models.png" width="100%">

### Public Datasets Performance
We have also performed evaluation on selected public datasets for seven classes. It includes three English (MSP-Podcast v1.11 test1, IEMOCAP (avg of five test folds), and MELD (test split)) datasets, Chinese (M3ED test split), and Indonesian (IndoWaveSentiment, entire split) dataset. Note that MSP, IEMOCAP, and IndoWaveSentiment datasets were not included in the model training. The MERaLiON-SER model shows competitive performance for out-of-domain datasets.

<div style="text-align: center;">
    <img src="https://huggingface.co/MERaLiON/MERaLiON-SER-v1/resolve/main/EngEval_8models.png" width="80%" style="margin: auto; display: block;">
</div>

---
## ⚙️ Usage Examples

### 🔹 GPU Inference
```python
from transformers import AutoProcessor, AutoModelForAudioClassification
import torch, torchaudio

repo = "MERaLiON/MERaLiON-SER-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(repo)
model = AutoModelForAudioClassification.from_pretrained(repo, trust_remote_code=True).to(device)
model.eval()

# ------- Single wav exmple --------------
wav, sr = torchaudio.load("sample.wav")
if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
wav = torchaudio.transforms.Resample(sr, 16000)(wav)

inputs = processor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
with torch.inference_mode():
    out = model(**{k:v.to(device) for k,v in inputs.items() if k in ("input_features","attention_mask")})
logits, dims = out["logits"], out["dims"]

emo_idx = torch.argmax(logits, dim=1).item()
emo_map = ["Neutral","Happy","Sad","Angry","Fearful","Disgusted","Surprised"]
print("Predicted Emotion:", emo_map[emo_idx])
print("Valance/Arousal/Dominance:", dims.squeeze().tolist())

# -------- Batch inference example using above loaded wav file--------
wav = wav.squeeze().numpy() # tensor of size (41642,) --> (samples,)
wavs = [wav,wav,wav] # example list of wav (tensor) for batch, here batch size =3

batch_inputs = processor(
    wavs,                     # list of 1D numpy arrays
    sampling_rate=16000,
    return_tensors="pt",
     padding="max_length",
    return_attention_mask=True,
)
with torch.inference_mode():
    out = model(**{k: v.to(device) for k, v in batch_inputs.items() if k in ("input_features","attention_mask")})

logits, dims = out["logits"], out["dims"]  # logits: (B, 7), dims: (B, 3) where B is batch size
emo_ids = torch.argmax(logits, dim=1).tolist()

for i in range(len(wavs)):
    eid = emo_ids[i]
    vad = dims.tolist()[i]
    print(f"Batch Index {i} -> {emo_map[eid]} | VAD={vad}")
```

### 🔹 CPU Inference
```python
from transformers import AutoProcessor, AutoModelForAudioClassification
import torch, soundfile as sf, torchaudio

repo = "MERaLiON/MERaLiON-SER-v1"
processor = AutoProcessor.from_pretrained(repo)
model = AutoModelForAudioClassification.from_pretrained(repo, trust_remote_code=True).cpu().eval()

wav, sr = sf.read("sample.wav")
if wav.ndim > 1: wav = wav.mean(axis=1)
if sr != 16000:
    wav = torchaudio.functional.resample(torch.tensor(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()

inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
with torch.inference_mode():
    out = model(**inputs)
logits, dims = out["logits"], out["dims"]
emo_idx = torch.argmax(logits, dim=1).item()
emo_map = ["Neutral","Happy","Sad","Angry","Fearful","Disgusted","Surprised"]
print("Predicted Emotion:", emo_map[emo_idx])
print("Valance/Arousal/Dominance:", dims.squeeze().tolist())
```

## Citation

If you find this model/dataset/space useful in your research, please refer and cite the following papers:

```bibtex
@article{serv1,
  title={MERaLiON-SER: Robust Speech Emotion Recognition Model for English and SEA Languages},
  author={MERaLiON Team},  journal={http://arxiv.org/abs/2511.04914},
  year={2025}
}
@inproceedings{wang2025benchmarking,
  title={Benchmarking Contextual and Paralinguistic Reasoning in Speech-LLMs: A Case Study with In-the-Wild Data},
  author={Wang, Qiongqiong and Sailor, Hardik Bhupendra and Liu, Tianchi and Zhang, Wenyu and Huzaifah, Muhammad and Lertcheva, Nattadaporn and Sun, Shuo and Chen, Nancy F and Wu, Jinyang and Aw, AiTi},
  booktitle={Findings of EMNLP 2025},
  year={2025}
}
@inproceedings{cpqa_interspeech,
  title={Contextual Paralinguistic Data Creation for  Multi-Modal Speech-LLM: Data Condensation and Spoken {QA} Generation},
  author={Wang, Qiongqiong and Sailor, Hardik B and Liu, Tianchi and Aw, Ai Ti},
  booktitle={Proc. Interspeech},
  year={2025},
}

@inproceedings{cpqa_asru,
  title={Incorporating Contextual Paralinguistic
Understanding in Large Speech-Language Models},
  author={
      Wang, Qiongqiong and Sailor, Hardik B and Wong, Jeremy H. M. and Liu, Tianchi and Sun, Shuo and Zhang, Wenyu and Huzaifah, Muhammad and  Chen, Nancy and Aw, Ai Ti},
  booktitle={Proc. IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  year={2025},
}
