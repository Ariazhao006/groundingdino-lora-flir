
_base_ = ["cfg_flir_lora_stageb_s1_regdata_e4.py"]

# S2: keep S1 regularization/data changes and smooth tail LR further.
epochs = 4
lr = 1.4e-4
multi_step_lr = True
lr_drop_list = [2, 3]

# Slightly tighter stop for short-run tail control.
enable_early_stop = True
early_stop_patience = 4
early_stop_min_delta = 3e-4
early_stop_warmup_epochs = 1
