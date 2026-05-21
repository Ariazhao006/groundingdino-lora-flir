_base_ = ["cfg_flir_lora_stageb_plus8_gentle.py"]

# Exp3: objective strategy upgrade (loss/cost rebalance).
epochs = 4
use_ema = False
save_checkpoint_interval = 4

# Emphasize classification quality and matching confidence.
set_cost_class = 1.3
cls_loss_coef = 2.4
focal_alpha = 0.35
focal_gamma = 2.0

# Keep bbox terms unchanged to avoid destabilizing localization.
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

enable_early_stop = True
early_stop_patience = 5
early_stop_min_delta = 3e-4
early_stop_warmup_epochs = 2
