from .schemas import *

unet_orig_config = SupervisedTrainingConfig(
    model_type=ModelChoice.UnetOrig,
    optimizer=OptimizerChoice.AdamW,
    tile_size=256,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=32,
    learning_rate=1e-3,
    datasets=DatasetChoice.S1S2,
    mine_class_loss_weight=2.,
    loss_fn=BackpropLossChoice.DICE
)

unet_resblocks_config = SupervisedTrainingConfig(
    model_type=ModelChoice.UnetResBlocks,
    optimizer=OptimizerChoice.AdamW,
    tile_size=256,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=32,
    learning_rate=1e-3,
    datasets=DatasetChoice.S1S2,
    mine_class_loss_weight=2.,
    loss_fn=BackpropLossChoice.DICE
)

resnet18_unet_config = SupervisedTrainingConfig(
    model_type=ModelChoice.ResNet18UNet,
    optimizer=OptimizerChoice.AdamW,
    tile_size=256,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=64,
    learning_rate=1e-3,
    datasets=DatasetChoice.S1S2,
    mine_class_loss_weight=2.,
    loss_fn=BackpropLossChoice.DICE
)

resnet34_unet_config = SupervisedTrainingConfig(
    model_type=ModelChoice.ResNet34UNet,
    optimizer=OptimizerChoice.AdamW,
    tile_size=256,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=128,
    learning_rate=1e-3,
    datasets=DatasetChoice.S1S2,
    mine_class_loss_weight=2.,
    loss_fn=BackpropLossChoice.DICE
)

resnet50_unet_config = SupervisedTrainingConfig(
    model_type=ModelChoice.ResNet50UNet,
    optimizer=OptimizerChoice.AdamW,
    tile_size=256,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=128,
    learning_rate=5e-4,
    datasets=DatasetChoice.S1S2,
    mine_class_loss_weight=2.,
    loss_fn=BackpropLossChoice.DICE
)
