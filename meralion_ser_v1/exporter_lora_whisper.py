import re
import sys
import json
import torch
from pathlib import Path

from transformers import WhisperModel, WhisperProcessor
import torch
# These two must be your remote-code files in the same folder:
from configuration_ser_whisper_ecapa import SERConfig
from modeling_ser_whisper_ecapa import SERWhisperECAPA

# ---- User inputs ----
CKPT_PATH = sys.argv[1] if len(sys.argv) > 1 else "path/to/your_lightning.ckpt"
OUT_DIR   = sys.argv[2] if len(sys.argv) > 2 else "."

# ---------- helpers ----------
def strip_prefix(d, prefix="model."):
    if any(k.startswith(prefix) for k in d.keys()):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in d.items()}
    return d

def pick_any(iterable, default=None):
    for x in iterable:
        return x
    return default

def detect_lora_targets_and_r(enc_state):
    """
    Detect PEFT LoRA target modules (by scanning for *.lora_A.weight) and infer r.
    enc_state: keys relative to whisper.encoder.* (i.e., without "whisper.encoder." prefix)
    Returns (target_module_patterns, r) or (None, None) if not found.
    """
    lora_a_keys = [k for k in enc_state.keys() if k.endswith("lora_A.weight")]
    if not lora_a_keys:
        return None, None

    # target module path is everything up to ".lora_A.weight"
    module_names = sorted({re.sub(r"\.lora_A\.weight$", "", k) for k in lora_a_keys})
    # infer r from any lora_A.weight shape
    any_key = pick_any(lora_a_keys)
    r = enc_state[any_key].shape[0] if any_key else None

    # Convert absolute names to a wildcard pattern list PEFT understands.
    # PEFT supports regex-like patterns; we pass the plain module names (dot paths).
    # Example names: layers.0.self_attn.q_proj  -> these will match exactly.
    return module_names, r

# ---------- load ckpt ----------
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state = ckpt.get("state_dict", ckpt)
state = strip_prefix(state, "model.")

# Split whisper.* vs non-whisper (downstream/heads)
whisper_sd = {k: v for k, v in state.items() if k.startswith("whisper.")}
non_whisper_sd = {k: v for k, v in state.items() if not k.startswith("whisper.")}

# Keep only encoder part for LoRA reconstruction and strip "whisper.encoder."
enc_sd = {re.sub(r"^whisper\.encoder\.", "", k): v for k, v in whisper_sd.items() if k.startswith("whisper.encoder.")}

# ---------- build base whisper ----------
base_whisper = WhisperModel.from_pretrained("/data/projects/13003558/hardik/hf_cache/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e")
print(enc_sd)
# ---------- detect LoRA targets ----------
#targets, inferred_r = detect_lora_targets_and_r(enc_sd)
targets = ["q_proj", "v_proj", "k_proj"]
inferred_r = 16
if targets:
    print(f"[info] Detected LoRA on encoder modules ({len(targets)}):")
    for t in targets[:10]:
        print("   -", t)
    if len(targets) > 10:
        print(f"   ... (+{len(targets)-10} more)")
    print(f"[info] Inferred r = {inferred_r}")

    # Build a PEFT-wrapped encoder with these targets
    try:
        from peft import get_peft_model, LoraConfig
    except Exception as e:
        raise RuntimeError("PEFT is required to merge LoRA. `pip install peft` then rerun.") from e

    # If r is None, default to 16
    r = int(inferred_r) if inferred_r else 16

    # PEFT wants a list of module name patterns (exact names are fine)
    lcfg = LoraConfig(
        r=r,
        lora_alpha=16,
        lora_dropout=0.0,   # not used at inference
        bias="none",
        target_modules=targets,
        use_rslora=True
    )

    peft_enc = get_peft_model(base_whisper.encoder, lcfg)

    # Load encoder weights (including LoRA tensors) into peft-encoder
    missing, unexpected = peft_enc.load_state_dict(enc_sd, strict=False)
    print("[merge] encoder missing:", missing[:10], " ...")
    print("[merge] encoder unexpected:", unexpected[:10], " ...")

    # Merge LoRA -> standard encoder
    merged_encoder = peft_enc.merge_and_unload()
    base_whisper.encoder.load_state_dict(merged_encoder.state_dict(), strict=False)
else:
    print("[warn] No LoRA adapter tensors detected (no *.lora_A.weight found).")
    print("       Proceeding without PEFT merge; assuming encoder is plain base weights.")
    # Still update base with encoder weights from ckpt (if any)
    enc_only = {k[len("encoder."):] : v for k, v in whisper_sd.items() if k.startswith("whisper.encoder.")}
    base_whisper.encoder.load_state_dict(enc_only, strict=False)

# ---------- build HF SER model (remote code) ----------
# Load your config.json if present to keep id2label/label2id
cfg_path = Path("config.json")
if cfg_path.exists():
    cfg_dict = json.loads(cfg_path.read_text())
    config = SERConfig(**cfg_dict)
else:
    config = SERConfig(
        whisper_name="/data/projects/13003558/hardik/hf_cache/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e",
        num_emotions=7,
        pooling_type="hybrid",
        sampling_rate=16000,
        use_weighted_sum=False,
    )

hf_model = SERWhisperECAPA(config)

# 1) copy merged whisper weights first (avoid overwriting merged encoder later)
with torch.no_grad():
    hf_model.whisper.load_state_dict(base_whisper.state_dict(), strict=False)

# 2) load downstream + heads (non-whisper keys)
missing2, unexpected2 = hf_model.load_state_dict(non_whisper_sd, strict=False)
print("[load] downstream/heads missing:", missing2[:20], " ...")
print("[load] downstream/heads unexpected:", unexpected2[:20], " ...")

# ---------- save HF model + processor ----------
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
hf_model.save_pretrained(OUT_DIR)
WhisperProcessor.from_pretrained(config.whisper_name).save_pretrained(OUT_DIR)
print(f"[done] Saved merged model + processor to: {OUT_DIR}")
