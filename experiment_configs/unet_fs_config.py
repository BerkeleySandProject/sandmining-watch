from .schemas import *
from ml.augmentations import DEFAULT_AUGMENTATIONS
from os.path import expanduser

unet_fs_config = SupervisedTrainingConfig(
    model_type=ModelChoice.UnetSmall,
    tile_size=110,
    s2_channels=None,
    batch_size=64,
    learning_rate=3e-2,
    output_dir=expanduser("~/sandmining-watch/out/unet_small"),
    datasets=DatasetChoice.S1S2,
    augmentations=DEFAULT_AUGMENTATIONS
)

unet_resblocks_config = SupervisedTrainingConfig(
    model_type=ModelChoice.UnetResBlocks,
    tile_size=128,
    s2_channels=None,
    batch_size=64,
    learning_rate=3e-2,
    output_dir=expanduser("~/sandmining-watch/out/unet_resblocks"),
    datasets=DatasetChoice.S1S2,
    augmentations=DEFAULT_AUGMENTATIONS
)
