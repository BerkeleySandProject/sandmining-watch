import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
import satlaspretrain_models

from utils import (
    adjust_state_dict_prefix,
    swin_feature_extractor,
    download_model_weights,
)

weights_manager = satlaspretrain_models.Weights()

model_weights = {
    # AI2's model weights for Satlas with Swin-v2-Base architecture for single image multispectral paradigm
    "sentinel2_SwinB_SI_MS": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_ms.pth?download=true",
    # AI2's model weights for Satlas with Resnet152 architecture for single image multispectral paradigm
    "sentinel2_resnet152_si_ms": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_si_ms.pth?download=true",
}

# for model_checkpoint_id in model_weights.keys():
#     weights_url = model_weights[model_checkpoint_id]
#     download_model_weights(url=weights_url, save_dir="satlas_weights")
#
# satalas_swinbase = weights_manager.get_pretrained_model(
#     SATLAS_SWINBASE_SI_MS_CHECKPOINT_ID
# )
# satlas_resnet152_with_fpn = weights_manager.get_pretrained_model(
#     SATLAS_RESNET152_SI_MS_CHECKPOINT_ID, fpn=True
# )

model_name = "sentinel2_resnet152_si_ms"
models_list = list(model_weights.keys())
if model_name in models_list:
    if model_name == "sentinel2_resnet152_si_ms":
        model = torchvision.models.resnet152()
        weights_url = model_weights[model_name]
        download_model_weights(url=weights_url, save_dir="satlas_weights")
        state_dict = torch.load("./satlas_weights/sentinel2_resnet152_si_ms.pth")
        state_dict = adjust_state_dict_prefix(
            state_dict, "resnet", "backbone.resnet.", prefix_allowed_count=0
        )
        model.conv1 = torch.nn.Conv2d(
            9, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.load_state_dict(state_dict)
        cf = create_feature_extractor(
            model,
            {
                "relu": "x1",
                "layer1": "x2",
                "layer2": "x3",
                "layer3": "x4",
                "layer4": "out",
            },
        )
        output = cf(torch.rand(10, 9, 128, 128))

    elif model_name == "Sentinel2_SwinB_SI_MS":
        swin_model = torchvision.models.swin_v2_b()
        swin_state_dict = torch.load(
            "./satlas_weights/sentinel2_swinb_si_ms.pth?download=true"
        )
        swin_state_dict = adjust_state_dict_prefix(
            swin_state_dict, "backbone", "backbone.", 0
        )
        swin_model.features[0][0] = torch.nn.Conv2d(
            9, 128, kernel_size=4, stride=2, padding=3, bias=True
        )
        swin_model.load_state_dict(swin_state_dict)
        swin_cf = swin_feature_extractor(swin_model)

        feats = swin_cf(torch.rand(10, 9, 128, 128))
