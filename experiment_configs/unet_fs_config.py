from .schemas import *
from ml.augmentations import DEFAULT_AUGMENTATIONS
from os.path import expanduser

unet_fs_config = SupervisedTrainingConfig(
    model_type=ModelChoice.UnetSmall,
    tile_size=110,
    s2_channels=None,
    s2_norming=S2NormChoice.Div,
    batch_size=64,
    learning_rate=3e-2,
    output_dir=expanduser("~/sandmining-watch/out/unet_small"),
    datasets=DatasetChoice.S1S2,
    augmentations=DEFAULT_AUGMENTATIONS
)
