from .schemas import *
from os.path import expanduser

resnet50_moco_ft_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.ResNet50UNet,
    optimizer=OptimizerChoice.AdamW,
    tile_size=256,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.DivideBy10000,
    batch_size=256,
    learning_rate=1e-3,
    output_dir=expanduser("~/sandmining-watch/out/resnet50-moco"),
    datasets=DatasetChoice.S2_L1C,
    mine_class_loss_weight=2.,
    finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
    encoder_weights_path="/data/sand_mining/checkpoints/ssl4eo/B13_rn50_moco_0099.pth",
    loss_fn=BackpropLossChoice.DICE
)

resnet18_moco_ft_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.ResNet18UNet,
    optimizer=OptimizerChoice.AdamW,
    tile_size=256,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.DivideBy10000,
    batch_size=256,
    learning_rate=1e-3,
    output_dir=expanduser("~/sandmining-watch/out/resnet18-moco"),
    datasets=DatasetChoice.S2_L1C,
    mine_class_loss_weight=2.,
    finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
    encoder_weights_path="/data/sand_mining/checkpoints/ssl4eo/B13_rn18_moco_0099_ckpt.pth",
    loss_fn=BackpropLossChoice.DICE
)
