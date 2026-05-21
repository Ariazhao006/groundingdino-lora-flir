_base_ = ["cfg_flir_lora_plateau_c_sched_smooth.py"]

# Unified-v1.1 (E28): merges the full ema_e20 -> warmrestart_e24 -> plus8_gentle_rerun
# lineage into a single from-scratch run. Supersedes the earlier v1 (E26) design which
# only covered the first two stages.
#
# Per-stage best AP and best epoch from the 3-stage pipeline:
#   ema_e20:            regular_best=0.4244 @ ep13/20  (start: official groundingdino weights)
#   warmrestart_e24:    regular_best=0.4297 @ ep13/24  (start: ema_e20/best_regular)
#   plus8_gentle_rerun: regular_best=0.4318 @ ep2/8    (start: warmrestart_e24/best_regular)
#
# Three rules these runs share:
#   (a) Best AP was reached BEFORE any LR drop fired -- high-LR exploration is what gets the gains.
#   (b) EMA strictly hurts (ema_best=0.3837 vs regular_best=0.4244 on the 1572-sample subset).
#   (c) Each warm restart's value comes from RESETTING optimizer momentum + lowering peak LR,
#       not from training longer at a fixed LR. plus8 hit best at ep2/8 specifically because
#       its restart let the model relax into a slightly different basin within a few epochs.
#
# Translation to a single from-scratch run:
#   1. Long high-LR exploration window (ep 0-19, lr=2e-4). 19 effective high-LR epochs roughly
#      matches the cumulative high-LR-budget of the original ema_e20 stage (13 best-epoch) plus
#      the additional 6 epochs the two warm restarts effectively bought after ema_e20.
#   2. Late multi-step tail at [20, 25, 27]. Proportional pattern (71%/89%/96% of budget) is
#      more conservative than plus8's [3,6,7] in 8 ep (37.5/75/87.5%) because the budget is
#      longer here, so we keep the tail tighter to prevent over-decay before early stop fires.
#   3. EMA off (consistent with warmrestart and plus8 conclusions).
#   4. Early stop tuned to preserve plus8-style small late gains:
#        - min_delta = 5e-4   (between warmrestart's 1e-3 and plus8's 3e-4; 1e-3 would have
#                              thrown away plus8's +0.002 gain).
#        - patience  = 5      (plus8 used 5; gives one extra epoch of grace vs. warmrestart=4).
#        - warmup    = 12     (from-scratch climbs slower than warm-started runs; ema_e20 was
#                              still warming up around ep10 in its trajectory, so wait to ep12).
#   5. save_checkpoint_interval = 2 (inherited from C1) for optional post-hoc weight averaging
#      across the last few healthy checkpoints if needed.

epochs = 28
lr = 2e-4
multi_step_lr = True
lr_drop_list = [20, 25, 27]

use_ema = False

enable_early_stop = True
early_stop_patience = 5
early_stop_min_delta = 0.0005
early_stop_warmup_epochs = 12
