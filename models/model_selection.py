from enum import Enum

class ModelSelection(Enum):
    UnetSmall = 1
    UnetOrig = 2
    Segformer = 3

def get_model(
        selection: ModelSelection,
        n_channels,
        img_size, # Tuple (w x h)
        n_classes = 2,
        **kwargs
    ):
    if selection == ModelSelection.UnetSmall:
        from models.unet.unet_small import UNetSmall
        model = UNetSmall(n_channels, n_classes)
    elif selection == ModelSelection.UnetOrig:
        from models.unet.unet_original import UNet
        model = UNet(n_channels, n_classes)
    elif selection == ModelSelection.Segformer:
        from transformers import SegformerConfig
        from models.segformer.segformer_rv import SegformerForSemanticSegmentationForRV
        config = SegformerConfig(num_channels=n_channels, **kwargs)
        model = SegformerForSemanticSegmentationForRV(config, img_size=img_size)
    else:
        raise ValueError("Error in model selection")
    return model
