_base_ = ["cfg_flir_lora_plateau_c_sched_smooth.py"]

# C2: C1 + EMA for stabler late-epoch model selection.
use_ema = True
ema_decay = 0.9996
ema_epoch = 2
