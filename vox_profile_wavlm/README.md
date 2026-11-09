---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
license: openrail
language:
- en
metrics:
- accuracy
base_model:
- microsoft/wavlm-large
pipeline_tag: audio-classification
---
# WavLM-Large for Categorical Emotion Classification

# Model Description
This model includes the implementation of categorical emotion classification described in Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits (https://arxiv.org/pdf/2505.14648)

The training pipeline used is also the top performing solution (SAILER) in INTERSPEECH 2025 - Speech Emotion Challenge (https://lab-msp.com/MSP-Podcast_Competition/IS2025/). Note that we did not use all the augmentation and and did not use the transcript to make the model simple but still effective compared to our INTERSPEECH Challenge solution.
We use the MSP-Podcast data for training this model.


The included emotions are: 
<pre>
[
    'Anger', 
    'Contempt', 
    'Disgust', 
    'Fear', 
    'Happiness', 
    'Neutral', 
    'Sadness', 
    'Surprise', 
    'Other'
]
</pre>

- Library: https://github.com/tiantiaf0627/vox-profile-release

# How to use this model

## Download repo
```
git clone git@github.com:tiantiaf0627/vox-profile-release.git
```
## Install the package
```
conda create -n vox_profile python=3.8
cd vox-profile-release
pip install -e .
```

## Load the model
```python
# Load libraries
import torch
import torch.nn.functional as F
from src.model.emotion.wavlm_emotion import WavLMWrapper

# Find device
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# Load model from Huggingface
model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion").to(device)
model.eval()
```

## Prediction
```python
# Label List
emotion_list = [
    'Anger', 
    'Contempt', 
    'Disgust', 
    'Fear', 
    'Happiness', 
    'Neutral', 
    'Sadness', 
    'Surprise', 
    'Other'
]
    
# Load data, here just zeros as the example
# Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation)
# So you need to prepare your audio to a maximum of 15 seconds, 16kHz and mono channel
max_audio_length = 15 * 16000
data = torch.zeros([1, 16000]).float().to(device)[:, :max_audio_length]
logits, embeddings = model(data, return_feature=True)
    
# Probability and output
emotion_prob = F.softmax(logits, dim=1)
print(emotion_list[torch.argmax(emotion_prob).detach().cpu().item()])
```

## If you have any questions, please contact: Tiantian Feng (tiantiaf@usc.edu)

## Kindly cite our paper if you are using our model or find it useful in your work
```
@article{feng2025vox,
  title={Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits},
  author={Feng, Tiantian and Lee, Jihwan and Xu, Anfeng and Lee, Yoonjeong and Lertpetchpun, Thanathai and Shi, Xuan and Wang, Helin and Thebaud, Thomas and Moro-Velazquez, Laureano and Byrd, Dani and others},
  journal={arXiv preprint arXiv:2505.14648},
  year={2025}
}
```

Responsible use of the Model: the Model is released under Open RAIL license, and users should respect the privacy and consent of the data subjects, and adhere to the relevant laws and regulations in their jurisdictions in using our model.

❌ **Out-of-Scope Use**
- Clinical or diagnostic applications
- Surveillance
- Privacy-invasive applications
- No commercial use