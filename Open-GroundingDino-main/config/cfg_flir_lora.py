_base_ = ["cfg_odvg.py"]

# ---------- FLIR defaults ----------
batch_size = 2
epochs = 30
lr_drop = 20
lr = 2e-4
weight_decay = 0.01
clip_max_norm = 0.1
use_coco_eval = False

# ---------- Evaluation labels ----------
# Keep config pure-static for SLConfig dump compatibility.
label_list = ["person", "bicycle", "car", "dog"]

# ---------- LoRA switches ----------
use_lora = True
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05
lora_only_trainable = True
lora_also_train_keywords = []
lora_include_patterns = [
    "transformer.encoder.fusion_layers.*.attn.*",
    "transformer.encoder.layers.*.self_attn.*",
]
lora_target_linear_names = [
    "v_proj",
    "l_proj",
    "values_v_proj",
    "values_l_proj",
    "out_v_proj",
    "out_l_proj",
    "sampling_offsets",
    "attention_weights",
    "value_proj",
    "output_proj",
]
