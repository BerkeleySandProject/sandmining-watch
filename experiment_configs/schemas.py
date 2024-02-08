from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from rastervision.core.data import ClassConfig

class ModelChoice(Enum):
    Test = "test"
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
    SatmaeLargeVITDecoder = "satmae-large-vit-decoder"

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

class BackpropLossChoice(Enum):
    BCE = "BCE"
    DICE = "DICE"

class ThreeClassVariants(Enum):
    A = "a"
    B = "b"
    C = "c"

@dataclass
class SupervisedTrainingConfig:
    model_type: ModelChoice
    optimizer: OptimizerChoice
    # scheduler: SchedulerChoice # wip, not yet implemented
    tile_size: int
    s2_channels: Optional[List[int]] # If none, RV will take all channels
    s2_normalization: NormalizationS2Choice
    loss_fn: BackpropLossChoice
    batch_size: int
    learning_rate: float
    datasets: DatasetChoice
    mine_class_loss_weight: float

@dataclass
class ThreeClassSupervisedTrainingConfig:
    model_type: ModelChoice
    optimizer: OptimizerChoice
    # scheduler: SchedulerChoice # wip, not yet implemented
    tile_size: int
    s2_channels: Optional[List[int]] # If none, RV will take all channels
    s2_normalization: NormalizationS2Choice
    batch_size: int
    learning_rate: float
    datasets: DatasetChoice
    mine_class_loss_weight: float
    three_class_method: ThreeClassVariants
    low_confidence_weight: float

class FinetuningStratagyChoice(Enum):
    End2EndFinetuning = "end-2-end"  # Nothing is frozen
    LinearProbing = "linear-probing" # Encoder weights are frozen
    FreezeEmbed = "freeze-embed" # Only applicable for ViT! Patch embed layer is frozen.
    LoRA = "lora" # Low Rank Adaptation

@dataclass
class SupervisedFinetuningConfig(SupervisedTrainingConfig):
    finetuning_strategy: FinetuningStratagyChoice
    encoder_weights_path: Optional[bool]
    num_upsampling_layers: Optional[int] = None # Only applicable for SatMaeLargeDoubleUpsampling
    apply_smoothing: Optional[bool] = True
    smoothing_sigma: Optional[float] = 10.

@dataclass
class InferenceConfig(SupervisedTrainingConfig):
    crop_sz: Optional[int]
    encoder_weights_path: Optional[bool]
    num_upsampling_layers: Optional[int] = None # Only applicable for SatMaeLargeDoubleUpsampling
    apply_smoothing: Optional[bool] = True
    smoothing_sigma: Optional[float] = 10.
    wandb_id: Optional[str] = None
    mean_threshold: Optional[float] = None


## Annotation stuff
class AnnotationType(Enum):
    TWO_ClASS = "2class"
    THREE_CLASS = "3class"

@dataclass
class AnnotationConfig:
    type: AnnotationType
    class_config: ClassConfig
    num_classes: int
    labelbox_project_id: str
    postfix: Optional[str] = None # needed when finding the GCP annotations path. by default 2_class doesnt need this

