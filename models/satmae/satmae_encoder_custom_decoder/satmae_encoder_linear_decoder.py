import torch.nn as nn
from einops import rearrange

from ..satmae_pretrained_encoder import SatMaePretrained
from ..pretrained_satmae_config import PATCH_SIZE, INPUT_SIZE


class SatMaeSegmenterWithLinearDecoder(SatMaePretrained):
    def __init__(self, vit_size):
        super().__init__(vit_size)
        self.decoder = nn.Linear(
            self.encoder_real_depth, 2,
        )

    def forward(self, x):
        x = self.encoder(x)
        
        # remove CLS/DIST tokens for decoding
        x = x[:, 1:]  # (N, gL, D)

        # g is the number of channel groups
        x = rearrange(x, 'N (g L) D -> N L (g D)', g=self.n_channel_groups)

        logits = self.decoder(x)

        # In the following, h = H/P and w = W/P
        # where H is the height in the input image and P is the height and widt of each patch
        logits = rearrange(logits, "N (h w) gD -> N gD h w", h=self.n_patches_along_axis)

        upsampled_logits = nn.functional.interpolate(
            logits, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
        )
        return upsampled_logits
    