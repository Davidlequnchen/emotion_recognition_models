# Emotion recognition models

Versioned model files for [emotion_recognition](https://github.com/Davidlequnchen/emotion_recognition).
This repository contains all eight models' original cards, configurations, tokenizers, required offline model
code, licences, and a checksum manifest. Weight bytes are GitHub Release attachments, not Git or Git LFS objects.

## Download

Python 3.9+ is the only requirement for the downloader. No Hugging Face, ModelScope, Git LFS, PyTorch or account
login is needed to download the public release.

```bash
python3 download.py                 # all eight models (64.00 GB of weights)
python3 download.py --list          # list models without downloading
python3 download.py melati_emotion  # download only one model
python3 download.py --verify-only   # verify existing weights, without network access
```

The downloader skips valid existing files, resumes interrupted transfers, verifies each attachment and the
reassembled file using SHA-256, and replaces a target only after the full verification succeeds. The attachment
cache is removed after each successful weight file; `--keep-cache` retains it. `--workers 4` downloads up to four
parts of the current weight in parallel. All weights require about 64 GB of final disk space; allow about 16 GB
of additional space while assembling the largest file. Model dependencies are installed separately by the main
project.

After a new clone of the main project:

```bash
git clone --recurse-submodules https://github.com/Davidlequnchen/emotion_recognition.git
cd emotion_recognition
python3 emotion_recognition/scripts/common/setup_models.py
```

A normal `git clone` does not initialize submodules; the setup script initializes this model submodule when
needed. Git does not download Release attachments automatically: the final Python command downloads them.
Existing machines should `git pull` and run the same setup command. A leftover local `models/` directory is
preserved as a backup and verified weights are reused, so they are not downloaded again.

## Included models

| Folder | Weight size (GB) | Publisher / source | Terms |
|---|---:|---|---|
| `emotion2vec_plus_large/` | 1.946 | [emotion2vec+ large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) | FunASR model licence; attribution and model name retained |
| `meralion_ser_v1/` | 3.080 | [MERaLiON-SER-v1](https://huggingface.co/MERaLiON/MERaLiON-SER-v1) | MERaLiON Public Licence SER V1 (original PDF included) |
| `melati_emotion/` | 1.275 | [Melati](https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3) | Non-commercial research only (original LICENSE included) |
| `vox_profile_whisper/` | 1.286 | [Vox-Profile Whisper slim](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion) | Publisher OpenRAIL declaration and non-commercial/use restrictions in model card |
| `vox_profile_wavlm/` | 1.265 | [Vox-Profile WavLM](https://huggingface.co/tiantiaf/wavlm-large-categorical-emotion) | Publisher OpenRAIL declaration and non-commercial/use restrictions in model card |
| `qwen2_5_omni_7b/` | 22.363 | [Qwen2.5-Omni 7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | Apache-2.0 |
| `qwen2_audio_7b/` | 16.794 | [Qwen2-Audio 7B Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | Apache-2.0 |
| `gemma4_e4b/` | 15.993 | [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) | Apache-2.0 |

Sizes use decimal GB. Each model retains its own source terms; no single licence is applied to all models.
Original cards and notices are preserved. `LICENSES.json` records licence sources and identifies licence text
added by this mirror when an upstream repository did not ship a standalone copy.

## Release and provenance

The fixed release tag is `models-2026-11-09-v1`. `manifest.json` pins the release, upstream revisions, original
weight names, sizes, full-file SHA-256 values, and the ordered attachments with their individual checksums.
Attachments are at most 1.5 GiB each, below GitHub's 2 GiB limit. The release is a mirror of files used by the
research project, not a new model trained by this repository owner.

- Weight bytes are unchanged by splitting and reconstruction. The original Qwen shard names and indexes are retained.
- Vox-Profile Whisper uses the project's existing `model_slim.safetensors`; the unused original 6.19 GB file is
  omitted. The slim file removes the unused text decoder and exactly represents frozen encoder values in float16;
  its own safetensors metadata records the original source and checksums. The existing prediction head remains intact.
- Offline base-model configurations already prepared by the research project are included for MERaLiON and
  Vox-Profile; emotion2vec's existing project configuration is retained.
- Qwen2.5-Omni retains all five weight shards needed by the current loader; its unused `spk_dict.pt` speech-output
  voices are omitted, as configured by the main project.
- Download caches, compiled files, previous 90 MiB Git-transfer parts, and redundant original Vox-Profile Whisper
  weights are omitted. The original files on the maintainer's machine are preserved.

The main repository pins this repository as the `emotion_recognition/models` submodule. Updating model versions
requires an explicit new manifest/release and submodule revision; ordinary `git pull` does not replace weights or
silently select the latest upstream release.

## Downloader tests

Run `python3 -m unittest discover -s tests`. Tests use a local HTTP server and tiny fixtures; no model downloads are needed.
