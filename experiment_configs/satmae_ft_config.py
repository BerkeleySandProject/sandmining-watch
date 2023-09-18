from .schemas import *
from models.satmae.pretrained_satmae_config import satmea_pretrained_encoder_bands_idx
from ml.augmentations import DEFAULT_AUGMENTATIONS
from os.path import expanduser

satmae_ft_lineardecoder_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.SatmaeBaseLinearDecoder,
    tile_size=96,
    s2_channels=satmea_pretrained_encoder_bands_idx,
    batch_size=64,
    learning_rate=3e-2,
    output_dir=expanduser("~/sandmining-watch/out/satmae-ft"),
    datasets=DatasetChoice.S2,
    augmentations=DEFAULT_AUGMENTATIONS,
    mine_class_loss_weight=2.,
    freeze_encoder_weights=True,
    encoder_weights_path=None # "/data/sand_mining/checkpoints/satmae_orig/pretrain-vit-base-e199.pth"
)

satmae_ft_doubleupsampling_config = SupervisedFinetuningCofig(
    model_type=ModelChoice.SatmaeBaseDoubleUpsampling,
    tile_size=96,
    s2_channels=satmea_pretrained_encoder_bands_idx,
    batch_size=64,
    learning_rate=3e-2,
    output_dir=expanduser("~/sandmining-watch/out/satmae-ft-doubleupsampling"),
    datasets=DatasetChoice.S2,
    augmentations=DEFAULT_AUGMENTATIONS,
    mine_class_loss_weight=2.,
    freeze_encoder_weights=True,
    encoder_weights_path="/data/sand_mining/checkpoints/satmae_orig/pretrain-vit-base-e199.pth"
)
