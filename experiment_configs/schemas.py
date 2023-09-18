from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

import albumentations as A


class ModelChoice(Enum):
    UnetSmall = "unet-small"
    UnetOrig = "unet-orig"
    Segformer = "segformer"
    SatmaeBaseLinearDecoder = "satmae-base-linear-decoder"
    SatmaeBaseDoubleUpsampling = "satmae-base-double-upsampling"
    UnetResBlocks = "unet-res-blocks"


class DatasetChoice(Enum):
    S2 = "s2" # Only S2 data
    S1S2 = "s1s2"  # S1 and S2 data


@dataclass
class SupervisedTrainingConfig:
    model_type: ModelChoice
    tile_size: int
    s2_channels: Optional[List[int]] # If none, RV will take all channels
    batch_size: int
    learning_rate: float
    output_dir: str
    datasets: DatasetChoice
    augmentations: A.Compose
    mine_class_loss_weight: float

@dataclass
class SupervisedFinetuningCofig(SupervisedTrainingConfig):
    freeze_encoder_weights: bool # (Technically, if this set to true, we are linear probing and not finetuning)

    # Sometimes, we will resume a finetuning job. In this case, we don't load the pretrained encoder weights,
    # but the checkpoint from finetuning. Therefore, encoder_weights_path is an optional parameter.
    encoder_weights_path: Optional[bool]
