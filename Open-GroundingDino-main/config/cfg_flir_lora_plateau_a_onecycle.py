_base_ = ["cfg_flir_lora.py"]

# Scheme A: smooth LR schedule to reduce early plateau.
# Uses OneCycleLR implemented in main.py.
batch_size = 2
epochs = 15
lr = 2e-4
onecyclelr = True
multi_step_lr = False

# Keep this large so no extra "checkpointXXXX" is dumped by lr_drop trigger.
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
