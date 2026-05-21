_base_ = ["cfg_odvg.py"]

# Zero-epoch evaluation config:
# evaluate pretrained model directly on FLIR val set.
use_lora = False
use_ema = False
use_checkpoint = False
use_transformer_ckpt = False
use_coco_eval = False

# Keep FLIR label space consistent with your experiments.
label_list = ["person", "bicycle", "car", "dog"]
