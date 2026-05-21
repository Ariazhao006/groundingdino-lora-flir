_base_ = ["cfg_flir_lora.py"]

# Scheme B: delayed decay + two-stage style behavior via milestones.
# Stage-1 keeps LR high longer, stage-2 decays later to continue refinement.
batch_size = 2
epochs = 15
lr = 2e-4
onecyclelr = False
multi_step_lr = True
lr_drop_list = [10, 13]

# Keep this large because scheduler is MultiStep; avoids extra checkpoint spam.
lr_drop = 100
save_checkpoint_interval = 100

# Keep LoRA setup aligned with baseline for fair comparison.
use_lora = True
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.05
lora_only_trainable = True

use_checkpoint = False
use_transformer_ckpt = False
