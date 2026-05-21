_base_ = ["cfg_flir_lora.py"]

# Quick run target: fit around ~2 hours on 1x4090.
batch_size = 2
epochs = 25
lr_drop = 4

# Keep LoRA setup from cfg_flir_lora.py
use_lora = True
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05

# Important for LoRA-only training:
# torch checkpointing can drop grad graph when most params are frozen.
use_checkpoint = False
use_transformer_ckpt = False
