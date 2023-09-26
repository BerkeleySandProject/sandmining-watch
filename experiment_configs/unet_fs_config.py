from .schemas import *
from ml.augmentations import DEFAULT_AUGMENTATIONS
from os.path import expanduser

unet_orig_config = SupervisedTrainingConfig(
    model_type=ModelChoice.UnetOrig,
    optimizer=OptimizerChoice.AdamW,
    tile_size=256,
    s2_channels=None,
    batch_size=32,
    learning_rate=1e-3,
    output_dir=expanduser("~/sandmining-watch/out/unet_small"),
    datasets=DatasetChoice.S1S2,
    augmentations=DEFAULT_AUGMENTATIONS,
    mine_class_loss_weight=2.,
)

unet_resblocks_config = SupervisedTrainingConfig(
    model_type=ModelChoice.UnetResBlocks,
    optimizer=OptimizerChoice.AdamW,
    tile_size=256,
    s2_channels=None,
    batch_size=32,
    learning_rate=1e-3,
    output_dir=expanduser("~/sandmining-watch/out/unet_resblocks"),
    datasets=DatasetChoice.S1S2,
    augmentations=DEFAULT_AUGMENTATIONS,
    mine_class_loss_weight=2.,
)

resnet18_unet_config = SupervisedTrainingConfig(
    model_type=ModelChoice.ResNet18UNet,
    tile_size=128,
    s2_channels=None,
    batch_size=128,
    learning_rate=3e-2,
    output_dir=expanduser("~/sandmining-watch/out/resnet18_unet"),
    datasets=DatasetChoice.S1S2,
    augmentations=DEFAULT_AUGMENTATIONS,
    mine_class_loss_weight=2.,
)

resnet34_unet_config = SupervisedTrainingConfig(
    model_type=ModelChoice.ResNet34UNet,
    tile_size=128,
    s2_channels=None,
    batch_size=128,
    learning_rate=3e-2,
    output_dir=expanduser("~/sandmining-watch/out/resnet34_unet"),
    datasets=DatasetChoice.S1S2,
    augmentations=DEFAULT_AUGMENTATIONS,
    mine_class_loss_weight=2.,
)

resnet50_unet_config = SupervisedTrainingConfig(
    model_type=ModelChoice.ResNet50UNet,
    tile_size=128,
    s2_channels=None,
    batch_size=128,
    learning_rate=3e-2,
    output_dir=expanduser("~/sandmining-watch/out/resnet50_unet"),
    datasets=DatasetChoice.S1S2,
    augmentations=DEFAULT_AUGMENTATIONS,
    mine_class_loss_weight=2.,
)
