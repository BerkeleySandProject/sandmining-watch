from enum import Enum

from experiment_configs.schemas import SupervisedTrainingConfig, SupervisedFinetuningConfig, ModelChoice, FinetuningStratagyChoice, InferenceConfig, ThreeClassConfig, ThreeClassSupervisedTrainingConfig, ThreeClassVariants
from typing import Union

from peft import LoraConfig, get_peft_model


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params/1e6}M || all params: {all_param/1e6}M || trainable%: {100 * trainable_params / all_param:.2f}")



def model_factory(
        config: Union[SupervisedTrainingConfig, SupervisedFinetuningConfig, ThreeClassConfig] ,
        n_channels,
        config_lora=None
    ):
    if isinstance(config, ThreeClassConfig) and \
        (config.three_class_training_method == ThreeClassVariants.B):
        n_classes = 3
    else:
        n_classes = 1
    if isinstance(config, SupervisedFinetuningConfig) and config.num_upsampling_layers is None:
        config.num_upsampling_layers = 2

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
        model = SatMaeSegmenterWithDoubleUpsampling("large", image_size=config.tile_size, num_classes=n_classes, num_levels=config.num_upsampling_layers)
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
    elif config.model_type == ModelChoice.Test:
        import torch
        model = torch.nn.Conv2d(n_channels, n_classes, 1)
    else:
        raise ValueError("Error in model selection")
    
    initial_state = {name: param.clone() for name, param in model.state_dict().items()}
    
    if isinstance(config, SupervisedFinetuningConfig):
        if config.encoder_weights_path:
            model.load_encoder_weights(config.encoder_weights_path)

        if config.finetuning_strategy == FinetuningStratagyChoice.LinearProbing or config.finetuning_strategy == FinetuningStratagyChoice.LoRA:
            model.freeze_encoder_weights()
        elif config.finetuning_strategy == FinetuningStratagyChoice.FreezeEmbed:
            model.freeze_embed_weights()
        elif config.finetuning_strategy == FinetuningStratagyChoice.LayerwiseLrDecay:
            raise NotImplementedError("LayerwiseLrDecay is not yet implemented")
        else:
            raise ValueError("Unknown choice for finetuning strategy")
        
    if isinstance(config, InferenceConfig):
        if config.encoder_weights_path:
            model.load_encoder_weights(config.encoder_weights_path)
        else:
            raise ValueError("No encoder weights path provided for inference")

    import torch
    # Compare the initial and final state of the model
    count = 0
    for name, param in model.state_dict().items():
        if not torch.equal(initial_state[name], param):
            # print(f'Parameter {name} has changed')
            count += 1
    print(f'Number of parameters loaded: {count}')
    
    if isinstance(config, SupervisedFinetuningConfig) and not isinstance(config, InferenceConfig) and config.finetuning_strategy == FinetuningStratagyChoice.LoRA:
        if config_lora is None:
            raise ValueError("No lora config passed.")
        else:
            print ("Applying LoRA ...")
            model = get_peft_model(model, config_lora)
    
    return model
