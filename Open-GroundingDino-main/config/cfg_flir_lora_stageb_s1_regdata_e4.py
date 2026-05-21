_base_ = ["cfg_flir_lora_stageb_plus8_gentle.py"]

# S1: strategy-level regularization + data policy update.
# Keep short continuation window for fast A/B validation.
epochs = 4
use_ema = False

# Moderate regularization uplift for LoRA-only tuning.
weight_decay = 0.015
lora_dropout = 0.1
fusion_dropout = 0.05

# Small-object-friendly and moderate-strength augmentation.
data_aug_scales = [512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832]
data_aug_scales2_resize = [500, 600, 700]
data_aug_scales2_crop = [416, 640]

# Keep the same early-stop policy for fair A/B.
enable_early_stop = True
early_stop_patience = 5
early_stop_min_delta = 3e-4
early_stop_warmup_epochs = 2
