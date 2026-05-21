# dino_project 实验产物说明（按时间顺序）

这份文档是给你“快速回忆用”的版本：  
不讲复杂原理，重点讲三件事：

1. 每次训练跑完会生成什么文件、放在哪里、你该看哪个。
2. 现在 `logs` 里每个实验目录分别是做什么的（按最早到最新）。
3. 哪些文件可以考虑删除，哪些建议保留。

---

## 1) 每次跑完会输出什么？都在哪？

下面默认“实验目录”指的是你启动命令里的 `--output_dir`，例如：

- `/root/autodl-tmp/dino_project/Open-GroundingDino-main/logs/flir_lora_plateau_c_sched_smooth_ema_e20`

### A. 配置与运行记录（建议保留）

- `info.txt`  
  - 作用：完整运行日志（命令、参数、报错、关键提示）
  - 用途：排错第一入口

- `config_cfg.py`  
  - 作用：当次运行实际生效配置快照
  - 用途：复现实验时最有用

- `config_args_raw.json` / `config_args_all.json`  
  - 作用：CLI 参数原始值 / 合并后的完整参数
  - 用途：确认“这次到底是怎么跑的”

### B. 指标结果（分析必看）

- `log.txt`  
  - 作用：每个 epoch 一行指标（loss、AP、APS、时间等）
  - 用途：你平时做对比分析主要靠它

- `best_summary.json`  
  - 作用：汇总 best AP、best APS（regular/ema）
  - 用途：看最优结果时很方便

- `preferred_checkpoint.json`  
  - 作用：程序按规则给出的推荐模型（通常 regular vs ema 二选一）
  - 用途：推理时先看它推荐哪个 ckpt

### C. 模型文件（用于恢复训练/部署）

- `checkpoint.pth`  
  - 作用：最近一次训练状态（模型+优化器+调度器）
  - 用途：断点续训首选

- `checkpoint_best_regular.pth`  
  - 作用：regular 分支 best AP 模型
  - 用途：通常是主交付模型

- `checkpoint_best_ema.pth`（只有开 EMA 才会有）  
  - 作用：EMA 分支 best AP 模型
  - 用途：和 regular 对比后再决定是否使用

- `checkpoint_best_aps_regular.pth` / `checkpoint_best_aps_ema.pth`  
  - 作用：按 APS（小目标）最优保存
  - 用途：如果你只关心小目标，可优先看它

- `checkpointXXXX.pth`（如 `checkpoint0009.pth`）  
  - 作用：中间阶段快照
  - 用途：回溯中间 epoch 或做 checkpoint averaging

### D. 评估缓存（可选）

- `eval/latest.pth` 和 `eval/xxx.pth`  
  - 作用：COCO 评估对象缓存
  - 用途：深度复盘时有用，平时不常直接看

---

## 2) 当前 logs 里的实验（最早 -> 最新）

> 顺序按照我们实验推进的时间线写，便于你回忆。

### 1. `flir_lora_quick20`
- 目的：最早的 quick20 基线训练，验证能不能稳定跑起来。
- 特点：同目录里有多次重跑记录，属于“起步探索版”。

### 2. `flir_lora_quick20_debug`
- 目的：调试版 quick20（主要用来快速排查问题）。
- 特点：通常不是最终对比用实验。

### 3. `flir_lora_quick20_direct`
- 目的：quick20 的直接启动对照试验。
- 特点：用于验证启动方式/流程差异，不是最终主线版本。

### 4. `flir_lora_quick20_localbert`
- 目的：把文本编码器切到本地缓存路径，验证本地模型加载。

### 5. `flir_lora_quick20_localbert_fix`
- 目的：在 localbert 基础上的修正版（稳定性修复）。

### 6. `flir_lora_quick20_pretrain`
- 目的：加载预训练权重后的主基线实验（你后续大量对比都基于它）。
- 特点：这是“平台期问题”分析时最常引用的 baseline。

### 7. `flir_lora_plateau_a_onecycle_e15`
- 目的：方案 A，OneCycleLR，尝试解决平台期。
- 结果印象：有提升，但后段有一定回落。

### 8. `flir_lora_plateau_b_multistep_e15`
- 目的：方案 B，MultiStepLR 里程碑衰减。
- 结果印象：比 A 进一步提升峰值，但仍有回落。

### 9. `flir_lora_plateau_b_stable_e15`
- 目的：B 的“稳定版”，希望减少峰值后回撤。
- 结果印象：峰值不错，但回落问题依然明显。

### 10. `flir_lora_plateau_c_sched_smooth_e15`
- 目的：C1，只做“更平滑的 LR 调度”，先验证回落能否收敛。

### 11. `flir_lora_plateau_c_sched_smooth_ema_e15`
- 目的：C2，C1 + EMA（最初版本）。
- 结果印象：regular 表现很好，但当时 EMA 逻辑有问题，后续修过。

### 12. `flir_lora_plateau_c_sched_smooth_ema_e20`
- 目的：C2 延长到 20 epoch，观察是否继续涨点并保持稳定。
- 结果印象：best AP 再创新高（你目前这一轮的强结果之一）。

---

## 3) 哪些文件可以考虑删？（先保守后激进）

下面按风险分级：

## 低风险（优先可删）

这些删了通常不影响你后续结论和复现实验：

- 纯调试/早期探索目录中重复的中间文件：
  - `flir_lora_quick20_debug`
  - `flir_lora_quick20_direct`
  - `flir_lora_quick20_localbert`
  - `flir_lora_quick20_localbert_fix`
- 各实验目录中的 `eval/` 缓存文件（如果你不做深度误差复盘）
- 不再需要的 `checkpointXXXX.pth` 中间快照（保留 best 和 checkpoint.pth 即可）

## 中风险（看你是否还要复现）

- 旧实验目录里 `config_args_raw.json`、`config_args_all.json` 可删一部分（建议至少保留 `config_cfg.py` + `info.txt`）
- 已被更新实验替代的完整目录（例如若你确定只用 C2 系列）

## 高风险（建议先别删）

- 当前主线最优实验目录：
  - `flir_lora_plateau_c_sched_smooth_ema_e20`
  - `flir_lora_plateau_c_sched_smooth_ema_e15`
  - `flir_lora_quick20_pretrain`（baseline 对照非常重要）
- 以下关键文件建议长期保留：
  - `log.txt`
  - `config_cfg.py`
  - `info.txt`
  - `checkpoint_best_regular.pth`（和你实际部署使用的模型）
  - `best_summary.json`
  - `preferred_checkpoint.json`

---

## 4) 一个实用保留模板（每个关键实验至少保留）

每个“你还要对比/复现”的实验目录，至少保留：

1. `log.txt`
2. `config_cfg.py`
3. `info.txt`
4. `checkpoint_best_regular.pth`（或你最终采用的 best 模型）
5. `best_summary.json` + `preferred_checkpoint.json`

这样基本能保证：  
你以后还能看懂结果、还能复现、还能继续接着改。

