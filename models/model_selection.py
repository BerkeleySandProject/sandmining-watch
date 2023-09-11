from experiment_configs.schemas import SupervisedTrainingConfig, SupervisedFinetuningCofig, ModelChoice
from typing import Union

def get_model(
        config: Union[SupervisedTrainingConfig, SupervisedFinetuningCofig] ,
        n_channels,
        n_classes = 2,
        **kwargs
    ):
    if config.model_type == ModelChoice.UnetSmall:
        from models.unet.unet_small import UNetSmall
        model = UNetSmall(n_channels, n_classes)
    elif config.model_type == ModelChoice.UnetOrig:
        from models.unet.unet_original import UNet
        model = UNet(n_channels, n_classes)
    elif config.model_type == ModelChoice.Segformer:
        from transformers import SegformerConfig
        from models.segformer.segformer_rv import SegformerForSemanticSegmentationForRV
        segformer_config = SegformerConfig(num_channels=n_channels, **kwargs)
        model = SegformerForSemanticSegmentationForRV(segformer_config, img_size=(config.tile_size, config.tile_size))
    elif config.model_type == ModelChoice.SatmaeBaseLinearDecoder:
        from models.satmae.satmae_encoder_custom_decoder.satmae_encoder_linear_decoder import SatMaeSegmenterWithLinearDecoder
        model = SatMaeSegmenterWithLinearDecoder()
    elif config.model_type == ModelChoice.SatmaeBaseDoubleUpsampling:
        from models.satmae.satmae_encoder_custom_decoder.satmae_encoder_double_upsampling import SatMaeSegmenterWithDoubleUpsampling
        model = SatMaeSegmenterWithDoubleUpsampling()
    else:
        raise ValueError("Error in model selection")
    
    if isinstance(config, SupervisedFinetuningCofig):
        if config.encoder_weights_path:
            model.load_encoder_weights(config.encoder_weights_path)
        if config.freeze_encoder_weights:
            model.freeze_encoder_weights()
    
    return model
