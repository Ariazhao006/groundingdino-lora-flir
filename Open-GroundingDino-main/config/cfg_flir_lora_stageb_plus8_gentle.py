_base_ = ["cfg_flir_lora_stageb_warmrestart_e24.py"]

# Continue from current best regular checkpoint for +8 epochs.
# Use gentler tail schedule and less aggressive early-stop threshold.
epochs = 8
lr = 1.6e-4
lr_drop_list = [3, 6, 7]

use_ema = False

enable_early_stop = True
early_stop_patience = 5
early_stop_min_delta = 3e-4
early_stop_warmup_epochs = 2
