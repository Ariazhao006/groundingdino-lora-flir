# Plateau Experiments (No Overwrite)

This plan creates two AP-focused experiments while preserving all existing runs.

## Safety Rules

- Never write to existing output dirs:
  - `logs/flir_lora_quick20`
  - `logs/flir_lora_quick20_pretrain`
- Use new output dirs only (listed below).
- Keep dataset and pretrain paths identical to baseline for fair comparison.

## New Configs

- Scheme A (smooth schedule): `config/cfg_flir_lora_plateau_a_onecycle.py`
- Scheme B (delayed decay): `config/cfg_flir_lora_plateau_b_multistep.py`
- Scheme B-stable (smoother delayed decay): `config/cfg_flir_lora_plateau_b_stable.py`

## Recommended Output Dirs

- A: `logs/flir_lora_plateau_a_onecycle_e15`
- B: `logs/flir_lora_plateau_b_multistep_e15`
- B-stable: `logs/flir_lora_plateau_b_stable_e15`

## Launch Commands (single GPU)

Run from `Open-GroundingDino-main`:

```bash
python main.py \
  --output_dir /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_a_onecycle_e15 \
  -c /root/autodl-tmp/dino_project/Open-GroundingDino-main/config/cfg_flir_lora_plateau_a_onecycle.py \
  --datasets /root/autodl-tmp/dino_project/Open-GroundingDino-main/config/datasets_flir_odvg_20p.json \
  --num_workers 6 \
  --pretrain_model_path /root/autodl-tmp/dino_project/weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=/root/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 \
  --amp
```

```bash
python main.py \
  --output_dir /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_b_multistep_e15 \
  -c /root/autodl-tmp/dino_project/Open-GroundingDino-main/config/cfg_flir_lora_plateau_b_multistep.py \
  --datasets /root/autodl-tmp/dino_project/Open-GroundingDino-main/config/datasets_flir_odvg_20p.json \
  --num_workers 6 \
  --pretrain_model_path /root/autodl-tmp/dino_project/weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=/root/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 \
  --amp
```

```bash
python main.py \
  --output_dir /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_b_stable_e15 \
  -c /root/autodl-tmp/dino_project/Open-GroundingDino-main/config/cfg_flir_lora_plateau_b_stable.py \
  --datasets /root/autodl-tmp/dino_project/Open-GroundingDino-main/config/datasets_flir_odvg_20p.json \
  --num_workers 6 \
  --pretrain_model_path /root/autodl-tmp/dino_project/weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=/root/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 \
  --amp
```

## Compare Against Baseline

- Baseline: `logs/flir_lora_quick20_pretrain/log.txt`
- New runs:
  - `logs/flir_lora_plateau_a_onecycle_e15/log.txt`
  - `logs/flir_lora_plateau_b_multistep_e15/log.txt`

Primary metric: `test_coco_eval_bbox[0]` (AP@[0.5:0.95]).

## Notes

- These configs intentionally reduce extra checkpoint files to avoid disk blow-up.
- `checkpoint.pth` and `checkpoint_best_regular.pth` are still saved for recovery.

## Rollback-Fix Batch (C/D)

New configs for plateau + rollback mitigation:

- C1 scheduler smoothing: `config/cfg_flir_lora_plateau_c_sched_smooth.py`
- C2 scheduler smoothing + EMA: `config/cfg_flir_lora_plateau_c_sched_smooth_ema.py`
- D1 C1 + stronger regularization: `config/cfg_flir_lora_plateau_d_reg_strong.py`
- D2 D1 + conservative tail LR: `config/cfg_flir_lora_plateau_d_reg_tail.py`

Recommended output dirs:

- C1: `logs/flir_lora_plateau_c_sched_smooth_e15`
- C2: `logs/flir_lora_plateau_c_sched_smooth_ema_e15`
- D1: `logs/flir_lora_plateau_d_reg_strong_e15`
- D2: `logs/flir_lora_plateau_d_reg_tail_e15`

### Unified stability metrics

Use the same indicators for every run:

- `best AP`
- `last AP`
- `last5 avg AP`
- `rollback = best AP - last AP`

Example:

```bash
python tools/analyze_training_stability.py \
  --logs \
  /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_c_sched_smooth_e15/log.txt \
  /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_c_sched_smooth_ema_e15/log.txt \
  /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_d_reg_strong_e15/log.txt \
  /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_d_reg_tail_e15/log.txt \
  --baseline-log /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_quick20_pretrain/log.txt \
  --save-json /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/rollback_fix_summary.json
```

### Checkpoint averaging (for stable delivery)

Average last-k or top-k checkpoints:

```bash
python tools/average_checkpoints.py \
  --output-dir /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_c_sched_smooth_ema_e15 \
  --mode top \
  --top-k 3 \
  --log /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_c_sched_smooth_ema_e15/log.txt \
  --key model \
  --save-path /root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_c_sched_smooth_ema_e15/checkpoint_avg_top3.pth
```
