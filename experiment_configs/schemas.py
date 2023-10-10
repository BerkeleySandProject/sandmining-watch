from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class ModelChoice(Enum):
    UnetSmall = "unet-small"
    UnetOrig = "unet-orig"
    Segformer = "segformer"
    SatmaeBaseLinearDecoder = "satmae-base-linear-decoder"
    SatmaeBaseDoubleUpsampling = "satmae-base-double-upsampling"
    SatmaeLargeDoubleUpsampling = "satmae-large-double-upsampling"
    UnetResBlocks = "unet-res-blocks"
    ResNet18UNet = "resnet18-unet"
    ResNet34UNet = "resnet34-unet"
    ResNet50UNet = "resnet50-unet"

class OptimizerChoice(Enum):
    AdamW = "adamw"
    SDG = "sdg"

class SchedulerChoice(Enum):
    Cyclic = "cyclic"

class DatasetChoice(Enum):
    # Which dataset(s) shall be uses
    S2 = "s2" # Only S2 (L2A) data
    S1S2 = "s1s2"  # S1 and S2 (L2A) data
    S2_L1C = "s2-l1c"

class NormalizationS2Choice(Enum):
    # Why normalization shall be applied on the S2 images
    ChannelWise = "channelwise" # For each channel, projects 4 standard deviations between [0,255] / [0,1]
    DivideBy10000 = "divideby10000"  # Used by SSL4EO

@dataclass
class SupervisedTrainingConfig:
    model_type: ModelChoice
    optimizer: OptimizerChoice
    # scheduler: SchedulerChoice # wip, not yet implemented
    tile_size: int
    s2_channels: Optional[List[int]] # If none, RV will take all channels
    s2_normalization: NormalizationS2Choice
    batch_size: int
    learning_rate: float
    output_dir: str
    datasets: DatasetChoice
    mine_class_loss_weight: float


class FinetuningStratagyChoice(Enum):
    End2EndFinetuning = "end-2-end"  # Nothing is frozen
    LinearProbing = "linear-probing" # Encoder weights are frozen
    FreezeEmbed = "freeze-embed" # Only applicable for ViT! Patch embed layer is frozen.
    LayerwiseLrDecay = "layerwise-lr-decay" # 

@dataclass
class SupervisedFinetuningCofig(SupervisedTrainingConfig):
    finetuning_strategy: FinetuningStratagyChoice

    # Sometimes, we will resume a finetuning job. In this case, we don't load the pretrained encoder weights,
    # but the checkpoint from finetuning. Therefore, encoder_weights_path is an optional parameter.
    encoder_weights_path: Optional[bool]
