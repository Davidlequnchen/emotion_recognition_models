---
license: other
license_name: model-license
license_link: https://github.com/alibaba-damo-academy/FunASR
frameworks:
- Pytorch
tasks:
- emotion-recognition
widgets:
  - enable: true
    version: 1
    task: emotion-recognition
    examples:
      - inputs:
          - data: git://example/test.wav
    inputs:
      - type: audio
        displayType: AudioUploader
        validator:
          max_size: 10M
        name: input
    output:
      displayType: Prediction
      displayValueMapping:
        labels: labels
        scores: scores
    inferencespec:
      cpu: 8
      gpu: 0
      gpu_memory: 0
      memory: 4096
    model_revision: master
    extendsParameters:
      extract_embedding: false
---

<div align="center">
    <h1>
    EMOTION2VEC+
    </h1>
    <p>
     emotion2vec+: speech emotion recognition foundation model <br>
    <b>emotion2vec+ large model</b>
    </p>
    <p>
    <img src="logo.png" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
</div>


# Guides
emotion2vec+ is a series of foundational models for speech emotion recognition (SER). We aim to train a "whisper" in the field of speech emotion recognition, overcoming the effects of language and recording environments through data-driven methods to achieve universal, robust emotion recognition capabilities. The performance of emotion2vec+ significantly exceeds other highly downloaded open-source models on Hugging Face.

![](emotion2vec+radar.png)

This version (emotion2vec_plus_large) uses a large-scale pseudo-labeled data for finetuning to obtain a large size model (~300M),  and currently supports the following categories:
0: angry
1: disgusted
2: fearful
3: happy
4: neutral
5: other
6: sad
7: surprised
8: unknown

# Model Card
GitHub Repo: [emotion2vec](https://github.com/ddlBoJack/emotion2vec)
|Model|⭐Model Scope|🤗Hugging Face|Fine-tuning Data (Hours)|
|:---:|:-------------:|:-----------:|:-------------:|
|emotion2vec|[Link](https://www.modelscope.cn/models/iic/emotion2vec_base/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_base)|/|
emotion2vec+ seed|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_seed/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_seed)|201|
emotion2vec+ base|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_base/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_base)|4788|
emotion2vec+ large|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_large)|42526|


# Data Iteration

We offer 3 versions of emotion2vec+, each derived from the data of its predecessor. If you need a model focusing on spech emotion representation, refer to [emotion2vec: universal speech emotion representation model](https://huggingface.co/emotion2vec/emotion2vec).

- emotion2vec+ seed: Fine-tuned with academic speech emotion data from [EmoBox](https://github.com/emo-box/EmoBox)
- emotion2vec+ base: Fine-tuned with filtered large-scale pseudo-labeled data to obtain the base size model (~90M)
- emotion2vec+ large: Fine-tuned with filtered large-scale pseudo-labeled data to obtain the large size model (~300M)

The iteration process is illustrated below, culminating in the training of the emotion2vec+ large model with 40k out of 160k hours of speech emotion data. Details of data engineering will be announced later. 

# Installation

`pip install -U funasr modelscope`

# Usage

input: 16k Hz speech recording

granularity:
- "utterance": Extract features from the entire utterance
- "frame": Extract frame-level features (50 Hz)

extract_embedding: Whether to extract features; set to False if using only the classification model

## Inference based on ModelScope

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_plus_large")

rec_result = inference_pipeline('https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav', granularity="utterance", extract_embedding=False)
print(rec_result)
```


## Inference based on FunASR

```python
from funasr import AutoModel

model = AutoModel(model="iic/emotion2vec_plus_large")

wav_file = f"{model.model_path}/example/test.wav"
res = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
print(res)
```
Note: The model will automatically download.


Supports input file list, wav.scp (Kaldi style):
```cat wav.scp
wav_name1 wav_path1.wav
wav_name2 wav_path2.wav
...
```

Outputs are emotion representation, saved in the output_dir in numpy format (can be loaded with np.load())

# Note

This repository is the Huggingface version of emotion2vec, with identical model parameters as the original model and Model Scope version. 

Original repository: [https://github.com/ddlBoJack/emotion2vec](https://github.com/ddlBoJack/emotion2vec)

Model Scope repository: [https://www.modelscope.cn/models/iic/emotion2vec_plus_large/summary](https://www.modelscope.cn/models/iic/emotion2vec_plus_large/summary)

Hugging Face repository: [https://huggingface.co/emotion2vec](https://huggingface.co/emotion2vec)

FunASR repository: [https://github.com/alibaba-damo-academy/FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/funasr1.0/examples/industrial_data_pretraining/emotion2vec)

# Citation
```BibTeX
@article{ma2023emotion2vec,
  title={emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation},
  author={Ma, Ziyang and Zheng, Zhisheng and Ye, Jiaxin and Li, Jinchao and Gao, Zhifu and Zhang, Shiliang and Chen, Xie},
  journal={arXiv preprint arXiv:2312.15185},
  year={2023}
}
```