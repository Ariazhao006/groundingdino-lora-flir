_base_ = ["cfg_flir_lora_stageb_plus8_gentle.py"]

# Exp2: capacity strategy upgrade (LoRA + light detection-head unfreeze).
epochs = 4
use_ema = False
save_checkpoint_interval = 4

# Train a small extra set of head params together with LoRA.
lora_only_trainable = True
lora_also_train_keywords = [
    "class_embed",
    "bbox_embed",
    "enc_out_class_embed",
    "enc_out_bbox_embed",
]

# Slightly lower LR because trainable params are increased.
lr = 1.2e-4
weight_decay = 0.01

enable_early_stop = True
early_stop_patience = 5
early_stop_min_delta = 3e-4
early_stop_warmup_epochs = 2
