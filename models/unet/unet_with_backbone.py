""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
import os
    
# https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/linear_BE_moco.py#L248-L276
# https://github.com/mberkay0/pretrained-backbones-unet/blob/main/backbones_unet/model/unet.py
# Download pretrained backbones from https://github.com/zhu-xlab/SSL4EO-S12
class ResNetEncoderUNetDecoder(nn.Module):
    def __init__(self, model_name, n_channels, n_classes) -> None:
        super().__init__()
        # Load TorchVision prebuilt model
        encoder_base = getattr(torchvision.models, model_name)()
        encoder_base.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove unnecessary end layers
        encoder_base.avgpool = nn.Sequential()
        encoder_base.flatten = nn.Sequential()
        encoder_base.fc = nn.Sequential()
        
        self.encoder = create_feature_extractor(encoder_base, {"relu": "x1", "layer1": "x2", "layer2": "x3", "layer3": "x4", "layer4": "out"})
        if model_name == "resnet18" or model_name == "resnet34":
            self.up1 = Up(512, 256, bilinear=False)
            self.up2 = Up(256, 128, bilinear=False)
            self.up3 = Up(128, 64, bilinear=False, up_in_channels=128)
            self.up4 = Up(128, 64, bilinear=False, up_in_channels=64)
        elif model_name == "resnet50" or model_name == "resnet101":
            self.up1 = Up(2048, 1024, bilinear=False)
            self.up2 = Up(1024, 512, bilinear=False)
            self.up3 = Up(512, 256, bilinear=False)
            self.up4 = Up(128, 64, bilinear=False, up_in_channels=256)
        else:
            raise ValueError("Resnet model_name is not supported")
        self.classifier = OutConv(64, n_classes)

    def load_encoder_weights(self, path_to_weights):
        if not os.path.isfile(path_to_weights):
            raise ValueError(f"No checkpoint found at {path_to_weights}")
        print(f"ResNetEncoderUNetDecoder: Loading encoder weights from {path_to_weights}")
        checkpoint = torch.load(path_to_weights, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        print(msg)

    def freeze_encoder_weights(self):
        print("ResNetEncoderUNetDecoder: Freezing encoder weights")
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        out = self.up1(features["out"], features["x4"])
        out = self.up2(out, features["x3"])
        out = self.up3(out, features["x2"])
        out = self.up4(out, features["x1"])
        out = self.classifier(out)
        out = F.interpolate(out, x.shape[2:], mode="bilinear", align_corners=False)
        return out
    