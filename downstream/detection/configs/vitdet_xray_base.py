"""Detectron2 LazyConfig base for SmartX X-ray detection."""

from functools import partial

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
dataloader = model_zoo.get_config("common/data/coco.py").dataloader
train = model_zoo.get_config("common/train.py").train
optimizer = model_zoo.get_config("common/optim.py").AdamW

model.roi_heads.mask_in_features = None
model.roi_heads.mask_pooler = None
model.roi_heads.mask_head = None

dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),
    L(T.ResizeScale)(
        min_scale=0.8,
        max_scale=1.2,
        target_height=768,
        target_width=768,
    ),
    L(T.FixedSizeCrop)(crop_size=(768, 768), pad=True),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.use_instance_mask = False
dataloader.train.mapper.recompute_boxes = False

dataloader.test.mapper.augmentations = [
    L(T.Resize)(shape=(768, 768)),
]
dataloader.test.mapper.image_format = "RGB"

optimizer.weight_decay = 0.1
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate,
    num_layers=24,
    lr_decay_rate=0.7,
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(start_value=1.0, end_value=0.01),
    warmup_length=0.05,
    warmup_factor=0.01,
)
