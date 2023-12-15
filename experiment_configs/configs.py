from .schemas import *

##########################
# Fully supervised

from project_config import S2Band
fully_supervised_band_selection = [
    S2Band.B2,
    S2Band.B3,
    S2Band.B4,
    S2Band.B8A,
    S2Band.B11,
    S2Band.B12,
] 
fully_supervised_band_selection_idx: list[int] = [
    e.value for e in fully_supervised_band_selection
]

unet_config = SupervisedTrainingConfig(
    model_type=ModelChoice.UnetOrig,
    optimizer=OptimizerChoice.AdamW,
    tile_size=160,
    s2_channels=fully_supervised_band_selection_idx,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=128,
    learning_rate=5e-4,
    datasets=DatasetChoice.S1S2,
    mine_class_loss_weight=6.,
    loss_fn=BackpropLossChoice.BCE
)

segformer_config = SupervisedTrainingConfig(
    model_type=ModelChoice.Segformer,
    optimizer=OptimizerChoice.AdamW,
    tile_size=160,
    s2_channels=fully_supervised_band_selection_idx,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=128,
    learning_rate=5e-4,
    datasets=DatasetChoice.S1S2,
    mine_class_loss_weight=6.,
    loss_fn=BackpropLossChoice.BCE
)

########################## 
# SSL4EO

ssl4eo_resnet18_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.ResNet18UNet,
    optimizer=OptimizerChoice.AdamW,
    tile_size=160,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.DivideBy10000,
    batch_size=128,
    learning_rate=5e-4,
    datasets=DatasetChoice.S2_L1C,
    mine_class_loss_weight=6.,
    finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
    encoder_weights_path="/data/sand_mining/checkpoints/ssl4eo/B13_rn18_moco_0099_ckpt.pth",
    loss_fn=BackpropLossChoice.BCE
)

ssl4eo_resnet50_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.ResNet50UNet,
    optimizer=OptimizerChoice.AdamW,
    tile_size=160,
    s2_channels=None,
    s2_normalization=NormalizationS2Choice.DivideBy10000,
    batch_size=128,
    learning_rate=5e-4,
    datasets=DatasetChoice.S2_L1C,
    mine_class_loss_weight=6,
    finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
    encoder_weights_path="/data/sand_mining/checkpoints/ssl4eo/B13_rn50_moco_0099.pth",
    loss_fn=BackpropLossChoice.BCE
)

##########################
# SatMAE

from models.satmae.pretrained_satmae_config import satmea_pretrained_encoder_bands_idx

satmae_base_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.SatmaeBaseDoubleUpsampling,
    optimizer=OptimizerChoice.AdamW,
    tile_size=160,
    s2_channels=satmea_pretrained_encoder_bands_idx,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=64,
    learning_rate=1e-3,
    datasets=DatasetChoice.S2,
    mine_class_loss_weight=6.,
    finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
    encoder_weights_path="/data/sand_mining/checkpoints/satmae_orig/pretrain-vit-base-e199.pth",
    loss_fn=BackpropLossChoice.BCE
)

satmae_large_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.SatmaeLargeDoubleUpsampling,
    optimizer=OptimizerChoice.AdamW,
    tile_size=160,
    s2_channels=satmea_pretrained_encoder_bands_idx,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=128,
    learning_rate=1e-3,
    datasets=DatasetChoice.S2,
    mine_class_loss_weight=6.,
    finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
    encoder_weights_path="/data/sand_mining/checkpoints/satmae_orig/pretrain-vit-large-e199.pth",
    loss_fn=BackpropLossChoice.BCE
)

## Inference

satmae_large_inf_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.SatmaeLargeDoubleUpsampling,
    optimizer=OptimizerChoice.AdamW,
    tile_size=200,
    s2_channels=satmea_pretrained_encoder_bands_idx,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=32,
    learning_rate=1e-3,
    datasets=DatasetChoice.S2,
    mine_class_loss_weight=0., #unused in inference mode
    finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
    encoder_weights_path="/data/sand_mining/checkpoints/finetuned/SatMAE-L_b128-BatchNormDec-smoothing.pth",
    loss_fn=BackpropLossChoice.BCE,
)
