_base_ = ["cfg_flir_lora_plateau_c_sched_smooth_ema.py"]

# C2-e20: extend C2 to 20 epochs for stability verification.
epochs = 20

# Re-space milestones for 20-epoch budget.
lr_drop_list = [14, 17, 19]
