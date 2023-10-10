from .schemas import *
from models.satmae.pretrained_satmae_config import satmea_pretrained_encoder_bands_idx
from os.path import expanduser

satmae_ft_lineardecoder_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.SatmaeBaseLinearDecoder,
    optimizer=OptimizerChoice.AdamW,
    tile_size=96,
    s2_channels=satmea_pretrained_encoder_bands_idx,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=256,
    learning_rate=1e-3,
    output_dir=expanduser("~/sandmining-watch/out/satmae-ft"),
    datasets=DatasetChoice.S2,
    mine_class_loss_weight=2.,
    finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
    encoder_weights_path=None # "/data/sand_mining/checkpoints/satmae_orig/pretrain-vit-base-e199.pth"
)

satmae_ft_doubleupsampling_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.SatmaeBaseDoubleUpsampling,
    optimizer=OptimizerChoice.AdamW,
    tile_size=96,
    s2_channels=satmea_pretrained_encoder_bands_idx,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=2,
    learning_rate=1e-3,
    output_dir=expanduser("~/sandmining-watch/out/satmae-ft-doubleupsampling"),
    datasets=DatasetChoice.S2,
    mine_class_loss_weight=2.,
    finetuning_strategy=FinetuningStratagyChoice.FreezeEmbed,
    encoder_weights_path="/data/sand_mining/checkpoints/satmae_orig/pretrain-vit-base-e199.pth"
)

satmae_large_ft_doubleupsampling_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.SatmaeLargeDoubleUpsampling,
    optimizer=OptimizerChoice.AdamW,
    tile_size=96,
    s2_channels=satmea_pretrained_encoder_bands_idx,
    s2_normalization=NormalizationS2Choice.ChannelWise,
    batch_size=64,
    learning_rate=1e-4,
    output_dir=expanduser("~/sandmining-watch/out/satmae-large-ft-doubleupsampling"),
    datasets=DatasetChoice.S2,
    mine_class_loss_weight=2.,
    finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
    encoder_weights_path="/data/sand_mining/checkpoints/satmae_orig/pretrain-vit-large-e199.pth"
)
