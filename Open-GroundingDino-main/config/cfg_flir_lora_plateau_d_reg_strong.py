_base_ = ["cfg_flir_lora_plateau_c_sched_smooth.py"]

# D1: keep smooth schedule, strengthen late-stage regularization via augmentation.
#
# Use wider resize scale range and larger crop branch to reduce overfitting.
data_aug_scales = [448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_scales2_resize = [384, 500, 640]
data_aug_scales2_crop = [384, 640]
