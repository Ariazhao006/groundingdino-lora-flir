_base_ = ["cfg_flir_lora_plateau_d_reg_strong.py"]

# D2: D1 + more conservative tail LR to reduce rollback after peak.
lr = 1.8e-4
lr_drop_list = [10, 12, 13, 14]
