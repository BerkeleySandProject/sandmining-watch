import torch
import torch.nn
import os
import ipdb

from .utils import (
    Backbone,
    Head,
    adjust_state_dict_prefix,
)
from .models.backbone import SwinBackbone
from .models.fpn import FPN, Upsample
from .models.heads import SimpleHead

import torch.nn.functional as F


class SatlasPretrained(torch.nn.Module):
    def __init__(
        self,
        num_channels=3,
        multi_image=False,
        backbone=Backbone.SWINB,
        fpn=False,
        head=None,
        num_categories=None,
        weights_path=None,
    ):
        """
        Initializes a model, based on desired imagery source and model components. This class can be used directly to
        create a randomly initialized model (if weights=None) or can be called from the Weights class to initialize a
        SatlasPretrain pretrained foundation model.

        Args:
            num_channels (int): Number of input channels that the backbone model should expect.
            multi_image (bool): Whether or not the model should expect single-image or multi-image input.
            backbone (Backbone): The architecture of the pretrained backbone. All image sources support SwinTransformer.
            fpn (bool): Whether or not to feed imagery through the pretrained Feature Pyramid Network after the backbone.
            head (Head): If specified, a randomly initialized head will be included in the model.
            num_categories (int): If a Head is being returned as part of the model, must specify how many outputs are wanted.
            weights (torch weights): Weights to be loaded into the model. Defaults to None (random initialization) unless
                                    initialized using the Weights class.
        """
        super(SatlasPretrained, self).__init__()

        # Validate user-provided arguments.
        if not isinstance(backbone, Backbone):
            raise ValueError("Invalid backbone.")
        if head and not isinstance(head, Head):
            raise ValueError("Invalid head.")
        if head and (num_categories is None):
            raise ValueError("Must specify num_categories if head is desired.")

        self.backbone = self._initialize_backbone(
            num_channels, backbone, multi_image, weights_path
        )

        ipdb.set_trace()
        # DEBUG: Ensure Feature Pyramid Network works properly
        if fpn:
            weights = torch.load(weights_path)
            self.fpn = self._initialize_fpn(
                self.backbone.out_channels, weights)
            self.upsample = Upsample(self.fpn.out_channels)
        else:
            self.fpn = None

        ipdb.set_trace()
        # DEBUG: Ensure segmentation head initializes properly
        if head is not None:
            self.head = (
                self._initialize_head(
                    head, self.fpn.out_channels, num_categories)
                if fpn
                else self._initialize_head(
                    head, self.backbone.out_channels, num_categories
                )
            )
        else:
            self.head = None

    def load_encoder_weights(self, path_to_weights):
        if not os.path.isfile(path_to_weights):
            raise ValueError(f"No checkpoint found at {path_to_weights}")
        print(f"Satlas: Loading encoder weights from {path_to_weights}")
        checkpoint = torch.load(path_to_weights, map_location="cpu")

        # check if checkpoint has a 'model' key
        # this indicates that you're loading the original Satlas pretrained weights

        # interpolate_pos_embed(self.encoder, checkpoint_model)
        # Load pretrained weights into the intialized backbone if weights were specified.
        prefix_allowed_count = 1
        if checkpoint is not None:
            state_dict = adjust_state_dict_prefix(
                checkpoint, "backbone", "backbone.", prefix_allowed_count
            )
            msg = self.backbone.load_state_dict(state_dict)

        if msg.missing_keys:
            print("Warning! Missing keys:")
            print(msg.missing_keys)
        # print(msg)

    def freeze_encoder_weights(self):
        print("SatlasPretrained: Freezing backbone/encoder weights")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _initialize_backbone(
        self, num_channels, backbone_arch, multi_image, weights_path
    ):
        # Load backbone model according to specified architecture.
        if backbone_arch == Backbone.SWINB:
            backbone = SwinBackbone(num_channels, arch="swinb")
        elif backbone_arch == Backbone.SWINT:
            backbone = SwinBackbone(num_channels, arch="swint")
        elif backbone_arch == Backbone.RESNET50:
            backbone = ResnetBackbone(num_channels, arch="resnet50")
        elif backbone_arch == Backbone.RESNET152:
            backbone = ResnetBackbone(num_channels, arch="resnet152")
        else:
            raise ValueError("Unsupported backbone architecture.")

        # If using a model for multi-image, need the Aggretation to wrap underlying backbone model.
        prefix, prefix_allowed_count = None, None
        if backbone_arch in [Backbone.RESNET50, Backbone.RESNET152]:
            prefix_allowed_count = 0
        elif multi_image:
            backbone = AggregationBackbone(num_channels, backbone)
            prefix_allowed_count = 2
        else:
            prefix_allowed_count = 1

        # Load pretrained weights into the intialized backbone if weights were specified.
        ipdb.set_trace()
        # DEBUG: Ensure weights are loaded and initialized properly
        if weights_path is not None:
            weights = torch.load(weights_path)
            state_dict = adjust_state_dict_prefix(
                weights, "backbone", "backbone.", prefix_allowed_count
            )
            backbone.load_state_dict(state_dict)

        return backbone

    def _initialize_fpn(self, backbone_channels, weights):
        fpn = FPN(backbone_channels)

        # Load pretrained weights into the intialized FPN if weights were specified.
        if weights is not None:
            state_dict = adjust_state_dict_prefix(
                weights, "fpn", "intermediates.0.", 0)
            fpn.load_state_dict(state_dict)
        return fpn

    def _initialize_head(self, head, backbone_channels, num_categories):
        # Initialize the head (classification, detection, etc.) if specified
        if head == Head.CLASSIFY:
            return SimpleHead("classification", backbone_channels, num_categories)
        elif head == Head.MULTICLASSIFY:
            return SimpleHead(
                "multi-label-classification", backbone_channels, num_categories
            )
        elif head == Head.SEGMENT:
            return SimpleHead("segment", backbone_channels, num_categories)
        elif head == Head.BINSEGMENT:
            return SimpleHead("bin_segment", backbone_channels, num_categories)
        elif head == Head.REGRESS:
            return SimpleHead("regress", backbone_channels, num_categories)
        elif head == Head.DETECT:
            return FRCNNHead("detect", backbone_channels, num_categories)
        elif head == Head.INSTANCE:
            return FRCNNHead("instance", backbone_channels, num_categories)
        return None

    def forward(self, imgs, targets=None):
        # Define forward pass. Implemented upsampling and argmax.

        x = self.backbone(imgs)
        import ipdb

        ipdb.set_trace()
        # DEBUG: Ensure forward pass works properly
        # if self.fpn == True:
        #     x = self.fpn(x)
        #     x = self.upsample(x)
        # if self.head:
        #     x, loss = self.head(imgs, x, targets)
        #     reshaped_tensor = x.view(32, 32, 3, 40, 40)
        #     upsampled_tensor = F.interpolate(
        #         reshaped_tensor, size=(160, 160), mode="bilinear", align_corners=False
        #     )
        #     segmentation_output = torch.argmax(upsampled_tensor, dim=1)
        #     return segmentation_output, loss
        x = self.backbone(x)
        x = self.upsample(x)
        x = self.head(x)
        x = x[0]
        x = torch.argmax(x, dim=1).unsqueeze(1).float()

        return x
