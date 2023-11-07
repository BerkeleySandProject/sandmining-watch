from .schemas import *

segformer_fs_config = SupervisedTrainingConfig(
    model_type=ModelChoice.Segformer,
    optimizer=OptimizerChoice.AdamW,
    tile_size=160,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=128,
    learning_rate=5e-4,
    datasets=DatasetChoice.S1S2,
    mine_class_loss_weight=6.,
    loss_fn=BackpropLossChoice.BCE
)
