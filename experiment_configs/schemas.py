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
    SatlasSwinBaseSI_MS_UnetDecoder = "satlas-swin-base-si-mis-unet-decoder"


class OptimizerChoice(Enum):
    AdamW = "adamw"
    SDG = "sdg"


class SchedulerChoice(Enum):
    Cyclic = "cyclic"


class DatasetChoice(Enum):
    # Which dataset(s) shall be uses
    S2 = "s2"  # Only S2 (L2A) data
    S1S2 = "s1s2"  # S1 and S2 (L2A) data
    S2_L1C = "s2-l1c"


class NormalizationS2Choice(Enum):
    # Why normalization shall be applied on the S2 images
    # For each channel, projects 4 standard deviations between [0,255] / [0,1]
    ChannelWise = "channelwise"
    DivideBy10000 = "divideby10000"  # Used by SSL4EO
    DivideBy255 = "divideby255"  # Used by Satlas


class BackpropLossChoice(Enum):
    BCE = "bce_loss"
    DICE = "dice_loss"


class ThreeClassVariants(Enum):
    A = "a"
    B = "b"


@dataclass
class SupervisedTrainingConfig:
    model_type: ModelChoice
    optimizer: OptimizerChoice
    # scheduler: SchedulerChoice # wip, not yet implemented
    tile_size: int
    s2_channels: Optional[List[int]]  # If none, RV will take all channels
    s2_normalization: NormalizationS2Choice
    loss_fn: BackpropLossChoice
    batch_size: int
    learning_rate: float
    datasets: DatasetChoice
    mine_class_loss_weight: float
    nonmine_class_weight: float
    uncertain_class_weight: float


@dataclass
class ThreeClassConfig:
    three_class_training_method: ThreeClassVariants
    low_confidence_weight: float


@dataclass
class ThreeClassSupervisedTrainingConfig(SupervisedTrainingConfig, ThreeClassConfig):
    pass


class FinetuningStratagyChoice(Enum):
    End2EndFinetuning = "end-2-end"  # Nothing is frozen
    LinearProbing = "linear-probing"  # Encoder weights are frozen
    # Only applicable for ViT! Patch embed layer is frozen.
    FreezeEmbed = "freeze-embed"
    LoRA = "lora"  # Low Rank Adaptation Linear Probing


@dataclass
class SupervisedFinetuningConfig(SupervisedTrainingConfig):
    finetuning_strategy: FinetuningStratagyChoice
    encoder_weights_path: Optional[bool]
    # Only applicable for SatMaeLargeDoubleUpsampling
    num_upsampling_layers: Optional[int] = None
    apply_smoothing: Optional[bool] = True
    smoothing_sigma: Optional[float] = 10.0


@dataclass
class ThreeClassFineTuningConfig(SupervisedFinetuningConfig, ThreeClassConfig):
    pass


@dataclass
class InferenceConfig(SupervisedTrainingConfig):
    crop_sz: Optional[int]
    encoder_weights_path: Optional[bool]
    # Only applicable for SatMaeLargeDoubleUpsampling
    num_upsampling_layers: Optional[int] = None
    apply_smoothing: Optional[bool] = True
    smoothing_sigma: Optional[float] = 10.0
    wandb_id: Optional[str] = None
    mean_threshold: Optional[float] = None


@dataclass
class ThreeClassInferenceConfig(InferenceConfig, ThreeClassConfig):
    pass


# Annotation stuff
class AnnotationType(Enum):
    TWO_ClASS = "2class"
    THREE_CLASS = "3class"


@dataclass
class AnnotationConfig:
    type: AnnotationType
    class_config: ClassConfig
    num_classes: int
    labelbox_project_id: str
    # needed when finding the GCP annotations path. by default 2_class doesnt need this
    postfix: Optional[str] = None
