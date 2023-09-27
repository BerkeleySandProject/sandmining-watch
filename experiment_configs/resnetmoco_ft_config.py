from .schemas import *
from ml.augmentations import DEFAULT_AUGMENTATIONS
from os.path import expanduser

resnet50_moco_ft_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.ResNet50UNet,
    optimizer=OptimizerChoice.AdamW,
    tile_size=96,
    s2_channels=None,
    batch_size=256,
    learning_rate=3e-2,
    output_dir=expanduser("~/sandmining-watch/out/resnet50-moco"),
    datasets=DatasetChoice.S2,  # TODO
    augmentations=DEFAULT_AUGMENTATIONS,
    mine_class_loss_weight=2.,
    freeze_encoder_weights=True,
    encoder_weights_path="/data/sand_mining/checkpoints/ssl4eo/B13_rn50_moco_0099.pth"
)
