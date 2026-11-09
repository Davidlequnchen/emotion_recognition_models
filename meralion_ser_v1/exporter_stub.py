
import torch, json
from configuration_ser_whisper_ecapa import SERConfig
from modeling_ser_whisper_ecapa import SERWhisperECAPA

# 1) Create config matching your training setup
config = SERConfig(
  whisper_name="openai/whisper-medium",
  num_emotions=7,
  pooling_type="hybrid",
  sampling_rate=16000,
  id2label={"0":"neutral","1":"happy","2":"sad","3":"angry","4":"fearful","5":"disgusted","6":"surprised"},
  label2id={"neutral":0,"happy":1,"sad":2,"angry":3,"fearful":4,"disgusted":5,"surprised":6},
)

# 2) Build model
model = SERWhisperECAPA(config)

# 3) Load Lightning checkpoint
ckpt = torch.load("path/to/your_lightning.ckpt", map_location="cpu")
state_dict = ckpt.get("state_dict", ckpt)

# Optionally map keys if they had prefixes like "model."
state_dict = {k.replace("model.", ""): v for k,v in state_dict.items()}

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing:", missing)
print("Unexpected:", unexpected)

# 4) Save HF format (writes pytorch_model.bin + config.json)
model.save_pretrained(".")

# 5) Save processor from the base whisper model so users can AutoProcessor.load from your repo
from transformers import WhisperProcessor
proc = WhisperProcessor.from_pretrained(config.whisper_name)
proc.save_pretrained(".")
print("Saved model + processor in current directory.")
