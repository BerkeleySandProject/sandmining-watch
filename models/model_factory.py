from enum import Enum

from experiment_configs.schemas import SupervisedTrainingConfig, SupervisedFinetuningCofig, ModelChoice, FinetuningStratagyChoice
from typing import Union


def model_factory(
        config: Union[SupervisedTrainingConfig, SupervisedFinetuningCofig] ,
        n_channels,
    ):
    n_classes = 1
    if config.model_type == ModelChoice.UnetSmall:
        from models.unet.unet_small import UNetSmall
        model = UNetSmall(n_channels, n_classes)
    elif config.model_type == ModelChoice.UnetOrig:
        from models.unet.unet_original import UNet
        model = UNet(n_channels, n_classes)
    elif config.model_type == ModelChoice.Segformer:
        from transformers import SegformerConfig
        from models.segformer.segformer_rv import SegformerForSemanticSegmentationForRV
        segformer_config = SegformerConfig(num_channels=n_channels, num_labels=n_classes) # SegFormer-B0
        model = SegformerForSemanticSegmentationForRV(segformer_config, img_size=config.tile_size)
    elif config.model_type == ModelChoice.SatmaeBaseLinearDecoder:
        from models.satmae.satmae_encoder_custom_decoder.satmae_encoder_linear_decoder import SatMaeSegmenterWithLinearDecoder
        model = SatMaeSegmenterWithLinearDecoder("base", image_size=config.tile_size, num_classes=n_classes)
    elif config.model_type == ModelChoice.SatmaeBaseDoubleUpsampling:
        from models.satmae.satmae_encoder_custom_decoder.satmae_encoder_double_upsampling import SatMaeSegmenterWithDoubleUpsampling
        model = SatMaeSegmenterWithDoubleUpsampling("base", image_size=config.tile_size, num_classes=n_classes)
    elif config.model_type == ModelChoice.SatmaeLargeDoubleUpsampling:
        from models.satmae.satmae_encoder_custom_decoder.satmae_encoder_double_upsampling import SatMaeSegmenterWithDoubleUpsampling
        model = SatMaeSegmenterWithDoubleUpsampling("large", image_size=config.tile_size, num_classes=n_classes)
    elif config.model_type == ModelChoice.UnetResBlocks:
        from models.unet.unet_resblocks import UNetResBlocks
        model = UNetResBlocks(n_channels, n_classes)
    elif config.model_type == ModelChoice.ResNet18UNet:
        from models.unet.unet_with_backbone import ResNetEncoderUNetDecoder
        model = ResNetEncoderUNetDecoder("resnet18", n_channels, n_classes)
    elif config.model_type == ModelChoice.ResNet34UNet:
        from models.unet.unet_with_backbone import ResNetEncoderUNetDecoder
        model = ResNetEncoderUNetDecoder("resnet34", n_channels, n_classes)
    elif config.model_type == ModelChoice.ResNet50UNet:
        from models.unet.unet_with_backbone import ResNetEncoderUNetDecoder
        model = ResNetEncoderUNetDecoder("resnet50", n_channels, n_classes)
    else:
        raise ValueError("Error in model selection")
    
    if isinstance(config, SupervisedFinetuningCofig):
        if config.encoder_weights_path:
            model.load_encoder_weights(config.encoder_weights_path)

        if config.finetuning_strategy == FinetuningStratagyChoice.End2EndFinetuning:
            pass
        elif config.finetuning_strategy == FinetuningStratagyChoice.LinearProbing:
            model.freeze_encoder_weights()
        elif config.finetuning_strategy == FinetuningStratagyChoice.FreezeEmbed:
            model.freeze_embed_weights()
        elif config.finetuning_strategy == FinetuningStratagyChoice.LayerwiseLrDecay:
            raise NotImplementedError("LayerwiseLrDecay is not yet implemented")
        else:
            raise ValueError("Unknown choise for finetuning strategy")
    
    return model
