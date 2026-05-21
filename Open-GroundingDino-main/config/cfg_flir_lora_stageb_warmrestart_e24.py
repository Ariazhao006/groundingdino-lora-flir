_base_ = ["cfg_flir_lora_plateau_c_sched_smooth.py"]

# Stage-B (warm restart):
# Start from previous best model weights via --pretrain_model_path
# and apply medium strategy updates without EMA.
epochs = 24
lr = 1.8e-4
multi_step_lr = True
lr_drop_list = [14, 18, 22]

# Disable EMA for this branch.
use_ema = False

# Early stopping on val AP to control rollback.
enable_early_stop = True
early_stop_patience = 4
early_stop_min_delta = 0.001
early_stop_warmup_epochs = 8
