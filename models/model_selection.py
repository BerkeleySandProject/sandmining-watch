from experiment_config import SupervisedTrainingConfig, ModelChoice

def get_model(
        config: SupervisedTrainingConfig,
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
    else:
        raise ValueError("Error in model selection")
    return model
