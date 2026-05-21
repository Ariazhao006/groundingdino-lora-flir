_base_ = ["cfg_flir_lora.py"]

# C1: smoother multi-step schedule to reduce post-peak rollback.
batch_size = 2
epochs = 15
lr = 2e-4
onecyclelr = False
multi_step_lr = True

# Compared with B/B-stable, spread decays to avoid sharp late transitions.
lr_drop_list = [11, 13, 14]

# Keep StepLR path inactive and save periodic checkpoints for averaging.
lr_drop = 100
save_checkpoint_interval = 2

# Important for LoRA-only training stability.
use_checkpoint = False
use_transformer_ckpt = False
