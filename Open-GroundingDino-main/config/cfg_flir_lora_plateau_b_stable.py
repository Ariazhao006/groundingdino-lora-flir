_base_ = ["cfg_flir_lora.py"]

# Scheme B-stable:
# Keep delayed decay idea from B, but smooth the late-stage transition
# to reduce post-peak rollback.
batch_size = 2
epochs = 15
lr = 1.8e-4
onecyclelr = False
multi_step_lr = True

# More gradual decays than B ([10, 13]) for better stability.
lr_drop_list = [9, 12, 14]

# Keep this large because scheduler is MultiStep; avoids extra checkpoint spam.
lr_drop = 100
save_checkpoint_interval = 100

# Keep LoRA setup aligned with baseline for fair comparison.
use_lora = True
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05
lora_only_trainable = True

# Important for LoRA-only training stability.
use_checkpoint = False
use_transformer_ckpt = False
