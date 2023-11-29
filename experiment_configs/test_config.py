from .schemas import *

test_config = SupervisedTrainingConfig(
    model_type=ModelChoice.Test,
    optimizer=OptimizerChoice.AdamW,
    tile_size=128,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=32,
    learning_rate=1e-3,
    datasets=DatasetChoice.S1S2,
    mine_class_loss_weight=2.,
    uncertain_class_weight=0.,
    loss_fn=BackpropLossChoice.BCE,
)
