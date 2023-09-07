from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import albumentations as A

from project_config import S2Band

class ModelChoice(Enum):
    UnetSmall = 1
    UnetOrig = 2
    Segformer = 3
    SatmaeLinearDecoder = 4

class S2NormChoice(Enum):
    SatMAE = "satmae"
    Div = "div"

class DatasetChoice(Enum):
    S2 = "s2"
    S1S2 = "s1s2"


@dataclass
class SupervisedTrainingConfig:
    model_type: ModelChoice
    tile_size: int
    s2_channels: Optional[List[int]] # If none, RV will take all channels
    s2_norming: S2NormChoice
    batch_size: int
    learning_rate: float
    output_dir: str
    datasets: DatasetChoice
    augmentations: A.Compose


DEFAULT_AUGMENTATIONS = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.CoarseDropout(max_height=32, max_width=32, max_holes=3)
])

## Example configuration to fine tune satmae
satmea_pretrained_encoder_bands = [
    # dropping B1
    S2Band.B2,
    S2Band.B3,
    S2Band.B4,
    S2Band.B5,
    S2Band.B6,
    S2Band.B7,
    S2Band.B8,
    S2Band.B8A,
    # dropping B9
    # dropping B10
    S2Band.B11,
    S2Band.B12,
]
satmea_pretrained_encoder_bands_idx: list[int] = [
    e.value for e in satmea_pretrained_encoder_bands
]

satmae_ft_config = SupervisedTrainingConfig(
    model_type=ModelChoice.SatmaeLinearDecoder,
    tile_size=96,
    s2_channels=satmea_pretrained_encoder_bands_idx,
    s2_norming=S2NormChoice.SatMAE,
    batch_size=64,
    learning_rate=3e-2,
    output_dir="out/satmae-ft",
    datasets=DatasetChoice.S2,
    augmentations=DEFAULT_AUGMENTATIONS
)


## Example configuration for Unet
unet_fs_config = SupervisedTrainingConfig(
    model_type=ModelChoice.UnetSmall,
    tile_size=110,
    s2_channels=None,
    s2_norming=S2NormChoice.Div,
    batch_size=64,
    learning_rate=3e-2,
    output_dir="out/unet_small",
    datasets=DatasetChoice.S1S2,
    augmentations=DEFAULT_AUGMENTATIONS
)
