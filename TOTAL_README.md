# Grounding DINO LoRA 微调（FLIR 红外数据集）

## 1. 项目简介

本项目对 **Grounding DINO**（开放词汇目标检测模型）进行 **LoRA 微调**，在 **FLIR ADAS 红外热成像数据集**上做目标检测实验。目标是用少量可训练参数（约占原模型 1%）让 Grounding DINO 适配到红外图像域。

### LoRA 注入位置

Grounding DINO 整体结构如下：

![Grounding DINO 结构](Open-GroundingDino-main/figs/grounding_dino_framework.png)

我们把 LoRA **只注入到图中第 2 部分 "Feature Enhancer Layer" 的所有注意力线性层**（同时也包含 encoder 的 self-attention 层）。主干（图像 backbone、文本 backbone）以及 Cross-Modality Decoder 全部冻结，**只有 LoRA 旁路参与训练**。

具体注入的线性层：

```
v_proj, l_proj, values_v_proj, values_l_proj,
out_v_proj, out_l_proj,
sampling_offsets, attention_weights, value_proj, output_proj
```

LoRA 超参：`rank=8, alpha=16, dropout=0.05`。

---

## 2. 代码结构

整体基于 [Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino) 框架（位于 `Open-GroundingDino-main/`），主要新增/修改的内容：

| 路径 | 说明 |
|------|------|
| `Open-GroundingDino-main/util/lora.py` | **新增**：LoRA 实现，包含 `LoRALinear`、按模块名注入函数、冻结函数 |
| `Open-GroundingDino-main/main.py` | **修改**：增加 LoRA 注入、参数冻结、early stopping 逻辑 |
| `Open-GroundingDino-main/config/cfg_flir_lora*.py` | **新增**：FLIR 实验所有配置文件（继承自基类 `cfg_flir_lora.py`） |
| `Open-GroundingDino-main/config/datasets_flir_odvg_20p.json` | **新增**：数据集路径配置（20% 训练子集 + 全量验证集） |
| `Open-GroundingDino-main/tools/flir_to_odvg.py` | **新增**：将 FLIR COCO 标注转 ODVG 格式的脚本 |
| `Open-GroundingDino-main/logs/` | 每次实验的训练日志（`log.txt`、`best_summary.json`），**不含模型权重** |

### 需要手动下载的内容（仓库未上传）

由于体积太大，以下三类文件没有放在仓库里，需要自行准备：

**(1) 预训练权重 `groundingdino_swint_ogc.pth`**：Grounding DINO 官方提供的 Swin-T 版本权重，是所有 LoRA 微调的起点。从 [GroundingDINO Releases](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha) 下载，放到：

```
dino_project/weights/groundingdino_swint_ogc.pth
```

**(2) FLIR ADAS 数据集**：训练 + 验证用的红外图像和标注。从 [FLIR ADAS 官网](https://www.flir.com/oem/adas/adas-dataset-form/) 申请下载 **FLIR ADAS 1.3** 版本，解压后放到：

```
dino_project/FLIR_ADAS_1_3/
├── train/      # 训练集原始 COCO 格式
├── val/        # 验证集原始 COCO 格式
└── video/      # （可选）
```

下载完成后，运行 `Open-GroundingDino-main/tools/flir_to_odvg.py` 把 COCO 标注转成 ODVG 格式，生成的文件应该放在 `dino_project/FLIR_ADAS_1_3/odvg/` 下（路径已在 `datasets_flir_odvg_20p.json` 中写死，可直接对应）：

```
FLIR_ADAS_1_3/odvg/
├── train_odvg_20p.jsonl      # 20% 训练子集的 ODVG 标注
├── val_coco_remapped.json    # 验证集 COCO 格式（4 类重映射）
└── label_map.json            # {"0":"person","1":"bicycle","2":"car","3":"dog"}
```

**(3) BERT 文本编码器**：Grounding DINO 用 `bert-base-uncased` 编码文本提示。第一次运行时会自动从 HuggingFace 下载并缓存到 `~/.cache/huggingface/`，**只要能访问 HuggingFace 就无需任何操作**。如果服务器不通外网，需要先手动下载 [bert-base-uncased](https://huggingface.co/bert-base-uncased)，再通过 `--options text_encoder_type=<本地路径>` 指定。本项目实际使用的本地缓存路径为：

```
~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594
```

---

## 3. Quick Start

### 环境配置

实测环境：**RTX 4090 / Python 3.8.10 / CUDA 11.8 / PyTorch 2.0.0+cu118**。

```bash
# 1) 创建并激活 Python 3.8 环境（用 conda 或 venv 都行）
conda create -n gdino python=3.8.10 -y
conda activate gdino

# 2) 安装 PyTorch（必须严格对应 cu118）
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 3) 安装其他依赖
cd dino_project/Open-GroundingDino-main
pip install -r requirements.txt

# 4) 编译 Deformable Attention 的 CUDA 算子（必须，否则训练会报错）
cd models/GroundingDINO/ops
python setup.py build install
cd ../../..
```

### 训练命令

确认上面三类文件都已经放好后，在 `Open-GroundingDino-main/` 目录下运行：

```bash
cd /root/autodl-tmp/dino_project/Open-GroundingDino-main

python -u main.py \
  --output_dir logs/flir_lora_plateau_c_sched_smooth_ema_e20 \
  -c config/cfg_flir_lora_plateau_c_sched_smooth_ema_e20.py \
  --datasets config/datasets_flir_odvg_20p.json \
  --num_workers 6 \
  --pretrain_model_path ../weights/groundingdino_swint_ogc.pth \
  --options text_encoder_type=~/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 \
  --amp \
  2>&1 | tee logs/flir_lora_ema_e20.out
```

参数说明：
- `--output_dir`：实验输出目录（日志、checkpoint、`best_summary.json` 都写在这）
- `-c`：实验配置文件，**替换它就可以跑不同实验**
- `--pretrain_model_path`：起点权重；第一阶段用官方预训练权重，后续阶段填上一阶段的 `checkpoint_best_regular.pth`
- `--amp`：混合精度，4090 必开
- `--options text_encoder_type=...`：如果能联网下载 BERT 可以删掉这一行

> 建议在 `tmux` 里跑，`Ctrl+b d` 断开后训练继续，`tmux attach` 重新连入查看进度。

---

## 4. 实验结果

**数据集**：FLIR ADAS 20% 子集（**1572 张训练图，1366 张验证图**），4 类：`person / bicycle / car / dog`。
**评估指标**：COCO 标准 AP（`AP@[0.50:0.95]` 为主指标）。

### 三阶段流水线（目前最优，AP=0.4318）

三段独立训练，每段把上一段的 `checkpoint_best_regular.pth` 作为下一段的起点（通过 `--pretrain_model_path` 加载，**只继承权重，不继承优化器状态**），这就是所谓的"warm restart"。

| 阶段 | 配置文件 | 起点 | lr | Epochs | 关键设置 | Best AP | Best Epoch |
|------|----------|------|----|--------|---------|---------|------------|
| Stage 1 | `cfg_flir_lora_plateau_c_sched_smooth_ema_e20.py` | 官方预训练权重 | 2e-4 | 20 | multi_step `[14,17,19]`，开 EMA | 0.4244 | 13 |
| Stage 2 | `cfg_flir_lora_stageb_warmrestart_e24.py` | Stage 1 best | 1.8e-4 | 24（早停@15） | multi_step `[14,18,22]`，关 EMA | 0.4297 | 13 |
| Stage 3 | `cfg_flir_lora_stageb_plus8_gentle.py` | Stage 2 best | 1.6e-4 | 8（早停@7） | multi_step `[3,6,7]`，关 EMA | **0.4318** | 2 |

**"最优"选取规则**：每个阶段内训练时，每跑完一个 epoch 都会在验证集上算 AP，只要刷新历史最高就自动覆盖保存为 `checkpoint_best_regular.pth`。三阶段下来 AP 从 `0.4244 → 0.4297 → 0.4318` 单调上涨，最终最优模型是 Stage 3 第 2 个 epoch 的 checkpoint。

> 文件位置：`Open-GroundingDino-main/logs/flir_lora_stageb_plus8_gentle_rerun/checkpoint_best_regular.pth`（未上传）。

### 一次跑通的对照实验（AP=0.4137）

把上面三阶段的 epoch 数和 LR 调度合并写进单个配置文件，**从官方预训练权重一次性训练 28 epoch**，验证三阶段的提升是否能被单次更长训练等效替代。

| 配置文件 | 起点 | lr | Epochs | 关键设置 | Best AP | Best Epoch |
|----------|------|----|--------|---------|---------|------------|
| `cfg_flir_lora_unified_v1_1_e28.py` | 官方预训练权重 | 2e-4 | 28（早停@17） | multi_step `[20,25,27]`，关 EMA，early_stop（patience=5, min_delta=5e-4, warmup=12） | 0.4137 | 10 |

结论：单次训练比三阶段流水线**低约 −0.018**。说明每阶段开始时"重置优化器动量 + 降低峰值 LR"带来的扰动是真实增益来源，单次训练（无论怎么调度）无法替代。

### 关键指标对比

| 指标 | 未微调 baseline | 三阶段最优 | 一次跑通 Stage 1 (ema_e20) |
|------|----------------|-----------|----------------------------|
| AP@[0.50:0.95] | 0.258 | **0.4318** | 0.4244 |
| AP50           | 0.490 | 0.7816    | 0.7565 |
| AP75           | 0.236 | 0.4410    | 0.4344 |
| APS / APM / APL | 0.141 / 0.349 / 0.591 | 0.2813 / 0.5202 / 0.7137 | 0.2789 / 0.5148 / 0.7088 |
| AR@1 / AR@10 / AR@100 | 0.142 / 0.407 / 0.454 | 0.2537 / 0.5771 / 0.6180 | 0.2468 / 0.5787 / 0.6231 |
| ARS / ARM / ARL | 0.297 / 0.541 / 0.791 | 0.4742 / 0.6857 / 0.8537 | 0.4985 / 0.6841 / 0.8611 |

LoRA 微调后 AP 从 `0.258` 提升到 `0.4318`（+0.174），证明 LoRA 注入 Feature Enhancer 是有效的低成本红外域适配方案。

---

## 5. 复刻指引

如果想从零复现一次最优实验，按顺序做：

1. 按"第 3 节 环境配置"装好环境；
2. 按"第 2 节"下载预训练权重、FLIR 数据集，跑 `tools/flir_to_odvg.py` 生成 ODVG 标注；
3. 用 `cfg_flir_lora_plateau_c_sched_smooth_ema_e20.py` 跑 **Stage 1**；
4. 跑完后把 `logs/flir_lora_plateau_c_sched_smooth_ema_e20/checkpoint_best_regular.pth` 作为 `--pretrain_model_path` 传入，用 `cfg_flir_lora_stageb_warmrestart_e24.py` 跑 **Stage 2**；
5. 同理用 Stage 2 的 best checkpoint 作为起点，用 `cfg_flir_lora_stageb_plus8_gentle.py` 跑 **Stage 3**；
6. 最终 best checkpoint 即为最优模型，对应 `best_summary.json` 中 `best_map.best_res ≈ 0.4318`。
