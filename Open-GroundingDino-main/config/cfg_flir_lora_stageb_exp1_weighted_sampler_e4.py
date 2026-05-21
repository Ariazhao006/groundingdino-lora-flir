_base_ = ["cfg_flir_lora_stageb_plus8_gentle.py"]

# Exp1: sampling strategy upgrade (small-object and dense-object upweighting).
epochs = 4
use_ema = False
save_checkpoint_interval = 4

# Enable weighted sampler (implemented in main.py).
train_sampler_mode = "weighted"
sampler_small_area_thr = 1600.0          # 40 * 40
sampler_small_object_boost = 1.0
sampler_box_count_boost = 0.35
sampler_box_count_ref = 6.0
sampler_min_weight = 1.0
sampler_max_weight = 2.8
sampler_replacement = True

# Keep early-stop logic consistent with recent stage-B runs.
enable_early_stop = True
early_stop_patience = 5
early_stop_min_delta = 3e-4
early_stop_warmup_epochs = 2
