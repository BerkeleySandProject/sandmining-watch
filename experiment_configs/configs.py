from peft import LoraConfig
from models.satmae.pretrained_satmae_config import satmea_pretrained_encoder_bands_idx
from models.satlas.pretrained_satlas_config import satlas_pretrained_encoder_bands
from .schemas import *

# There's no reason to set batch sizes to a powers of 2. Empirical evidence:
# https://wandb.ai/datenzauberai/Batch-Size-Testing/reports/Do-Batch-Sizes-Actually-Need-to-be-Powers-of-2---VmlldzoyMDkwNDQx

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

# # Used for Testing Only
# testing_config = SupervisedTrainingConfig(
#     model_type=ModelChoice.Test,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=fully_supervised_band_selection_idx,
#     s2_normalization=NormalizationS2Choice.ChannelWise,
#     batch_size=128,
#     learning_rate=5e-4,
#     datasets=DatasetChoice.S1S2,
#     mine_class_loss_weight=6.0,
#     loss_fn=BackpropLossChoice.BCE,
# )
#
# testing_configA = ThreeClassSupervisedTrainingConfig(
#     three_class_training_method=ThreeClassVariants.A,
#     low_confidence_weight=0.0,
#     **vars(testing_config),
# )
#
# testing_configB = ThreeClassSupervisedTrainingConfig(
#     three_class_training_method=ThreeClassVariants.B,
#     low_confidence_weight=1.0,
#     **vars(testing_config),
# )
#
# unet_config = SupervisedTrainingConfig(
#     model_type=ModelChoice.UnetOrig,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=fully_supervised_band_selection_idx,
#     s2_normalization=NormalizationS2Choice.ChannelWise,
#     batch_size=128,
#     learning_rate=5e-4,
#     datasets=DatasetChoice.S1S2,
#     mine_class_loss_weight=6.0,
#     loss_fn=BackpropLossChoice.BCE,
# )
#
# segformer_config = SupervisedTrainingConfig(
#     model_type=ModelChoice.Segformer,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=fully_supervised_band_selection_idx,
#     s2_normalization=NormalizationS2Choice.ChannelWise,
#     batch_size=128,
#     learning_rate=5e-4,
#     datasets=DatasetChoice.S1S2,
#     mine_class_loss_weight=6.0,
#     loss_fn=BackpropLossChoice.BCE,
# )
#
# ##########################
# # SSL4EO
#
# ssl4eo_resnet18_config = SupervisedFinetuningConfig(
#     model_type=ModelChoice.ResNet18UNet,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=None,
#     s2_normalization=NormalizationS2Choice.DivideBy10000,
#     batch_size=128,
#     learning_rate=5e-4,
#     datasets=DatasetChoice.S2_L1C,
#     mine_class_loss_weight=6.0,
#     finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
#     encoder_weights_path="/data/sand_mining/checkpoints/ssl4eo/B13_rn18_moco_0099_ckpt.pth",
#     loss_fn=BackpropLossChoice.BCE,
# )
#
# ssl4eo_resnet18_threeclass_configA = ThreeClassFineTuningConfig(
#     three_class_training_method=ThreeClassVariants.A,
#     low_confidence_weight=0.0,
#     **vars(ssl4eo_resnet18_config),
# )
#
# ssl4eo_resnet18_threeclass_configB = ThreeClassFineTuningConfig(
#     three_class_training_method=ThreeClassVariants.B,
#     low_confidence_weight=1.0,
#     **vars(ssl4eo_resnet18_config),
# )
#
# ssl4eo_resnet50_config = SupervisedFinetuningConfig(
#     model_type=ModelChoice.ResNet50UNet,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=None,
#     s2_normalization=NormalizationS2Choice.DivideBy10000,
#     batch_size=128,
#     learning_rate=5e-4,
#     datasets=DatasetChoice.S2_L1C,
#     mine_class_loss_weight=6,
#     finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
#     encoder_weights_path="/data/sand_mining/checkpoints/ssl4eo/B13_rn50_moco_0099.pth",
#     loss_fn=BackpropLossChoice.BCE,
# )

##########################
# SatMAE


# satmae_base_config = SupervisedFinetuningConfig(
#     model_type=ModelChoice.SatmaeBaseDoubleUpsampling,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=satmea_pretrained_encoder_bands_idx,
#     s2_normalization=NormalizationS2Choice.ChannelWise,
#     batch_size=64,
#     learning_rate=1e-3,
#     datasets=DatasetChoice.S2,
#     # mine_class_loss_weight=6.0,
#     nonmine_class_weight=3.0,
#     finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
#     encoder_weights_path="/data/sand_mining/checkpoints/satmae_orig/pretrain-vit-base-e199.pth",
#     loss_fn=BackpropLossChoice.BCE,
# )

# satmae_large_config = SupervisedFinetuningConfig(
#     model_type=ModelChoice.SatmaeLargeDoubleUpsampling,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=satmea_pretrained_encoder_bands_idx,
#     s2_normalization=NormalizationS2Choice.ChannelWise,
#     batch_size=100,
#     learning_rate=1e-3,
#     datasets=DatasetChoice.S2,
#     mine_class_loss_weight=6.0,
#     nonmine_class_weight=3.0,
#     finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
#     encoder_weights_path="/data/sand_mining/checkpoints/satmae_orig/pretrain-vit-large-e199.pth",
#     loss_fn=BackpropLossChoice.BCE,
#     num_upsampling_layers=2,
#     apply_smoothing=True,
#     smoothing_sigma=5.0,
# )
#
# satmae_large_config_lora = SupervisedFinetuningConfig(
#     model_type=ModelChoice.SatmaeLargeDoubleUpsampling,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=satmea_pretrained_encoder_bands_idx,
#     s2_normalization=NormalizationS2Choice.ChannelWise,
#     batch_size=8,
#     learning_rate=1e-3,
#     datasets=DatasetChoice.S2,
#     mine_class_loss_weight=6.0,
#     finetuning_strategy=FinetuningStratagyChoice.LoRA,
#     encoder_weights_path="/data/sand_mining/checkpoints/satmae_orig/pretrain-vit-large-e199.pth",
#     loss_fn=BackpropLossChoice.BCE,
#     num_upsampling_layers=2,
#     apply_smoothing=True,
#     smoothing_sigma=5.0,
# )
#
# satmae_large_config_lora_lp = SupervisedFinetuningConfig(
#     model_type=ModelChoice.SatmaeLargeDoubleUpsampling,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=satmea_pretrained_encoder_bands_idx,
#     s2_normalization=NormalizationS2Choice.ChannelWise,
#     batch_size=100,
#     learning_rate=1e-3,
#     datasets=DatasetChoice.S2,
#     mine_class_loss_weight=6.0,
#     finetuning_strategy=FinetuningStratagyChoice.LoRA,
#     encoder_weights_path="/home/ando/sandmining-watch/out/OUTPUT_DIR/SatMAE-L_LoRA-bias_LN_160px_mclw-6_B8_SmoothVal-E9.pth",
#     loss_fn=BackpropLossChoice.BCE,
#     num_upsampling_layers=2,
#     apply_smoothing=True,
#     smoothing_sigma=10.0,
# )

# satmae_large_config_lora_methodA = ThreeClassFineTuningConfig(
#     three_class_training_method=ThreeClassVariants.A,
#     low_confidence_weight=0.0,
#     **vars(satmae_large_config_lora),
# )
#
# satmae_large_config_lora_methodB = ThreeClassFineTuningConfig(
#     three_class_training_method=ThreeClassVariants.B,
#     low_confidence_weight=1.0,
#     **vars(satmae_large_config_lora),
# )

satlas_swin_base_si_ms_linear_decoder_config = ThreeClassFineTuningConfig(
    model_type=ModelChoice.SatlasSwinBaseSI_MS_UnetDecoder,
    optimizer=OptimizerChoice.AdamW,
    tile_size=128,
    s2_channels=satlas_pretrained_encoder_bands,
    s2_normalization=NormalizationS2Choice.DivideBy255,
    batch_size=32,
    learning_rate=1e-3,
    datasets=DatasetChoice.S2,
    mine_class_loss_weight=6.0,
    nonmine_class_weight=3.0,
    finetuning_strategy=FinetuningStratagyChoice.LinearProbing,
    encoder_weights_path="/data/sand_mining/checkpoints/satlas_orig/pretrain-swinb-si-ms.pth",
    loss_fn=BackpropLossChoice.BCE,
    num_upsampling_layers=None,
    apply_smoothing=True,
    smoothing_sigma=5.0,
    three_class_training_method=ThreeClassVariants.B,
    low_confidence_weight=1.0,
    num_channels=9,
    fpn=True,
)

# Inference

# satmae_large_inf_config = InferenceConfig(
#     model_type=ModelChoice.SatmaeLargeDoubleUpsampling,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=satmea_pretrained_encoder_bands_idx,
#     s2_normalization=NormalizationS2Choice.ChannelWise,
#     batch_size=128,
#     learning_rate=1e-3,
#     datasets=DatasetChoice.S2,
#     mine_class_loss_weight=0.0,  # unused in inference mode
#     encoder_weights_path="/data/sand_mining/checkpoints/finetuned/SatMAE-L_LoRA-bias_LN_160px_mclw-6_B8_E9_SmoothVal-S5-DecOnly-E20.pth",
#     loss_fn=BackpropLossChoice.BCE,
#     crop_sz=0,
#     apply_smoothing=True,
#     smoothing_sigma=5.0,
#     wandb_id="sandmining-watch/sandmine_detector/6r8ypwmb",
#     mean_threshold=0.51,
# )
#
# satmae_large_methodb_inf_config = ThreeClassInferenceConfig(
#     three_class_training_method=ThreeClassVariants.B,
#     low_confidence_weight=1.0,
#     **vars(satmae_large_inf_config),
# )
#
# satmae_large_methoda_inf_config = ThreeClassInferenceConfig(
#     three_class_training_method=ThreeClassVariants.A,
#     low_confidence_weight=0.0,
#     **vars(satmae_large_inf_config),
# )
#
# satmae_large_inf_config1 = InferenceConfig(
#     model_type=ModelChoice.SatmaeLargeDoubleUpsampling,
#     optimizer=OptimizerChoice.AdamW,
#     tile_size=160,
#     s2_channels=satmea_pretrained_encoder_bands_idx,
#     s2_normalization=NormalizationS2Choice.ChannelWise,
#     batch_size=128,
#     learning_rate=1e-3,
#     datasets=DatasetChoice.S2,
#     mine_class_loss_weight=0.0,  # unused in inference mode
#     encoder_weights_path="/data/sand_mining/checkpoints/finetuned/SatMAE-L_LoRA-bias_LN_160px_mclw-6_B8_SmoothVal_E-20.pth",
#     loss_fn=BackpropLossChoice.BCE,
#     crop_sz=0,
#     apply_smoothing=True,
#     smoothing_sigma=5.0,
#     wandb_id="sandmining-watch/sandmine_detector/mvuyz9n4",
#     mean_threshold=0.4336,
# )
#
#
# # LoRA Configs
# # Refer to https://huggingface.co/docs/peft/conceptual_guides/lora
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     target_modules=["qkv"],
#     lora_dropout=0.1,
#     # bias="none",
#     bias="lora_only",
#     # modules_to_save: List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include model’s custom head that is randomly initialized for the fine-tuning task.
#     modules_to_save=["decoder"],
# )
