import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..satmae_pretrained_encoder import SatMaePretrained
from ..pretrained_satmae_config import PATCH_SIZE, INPUT_SIZE


# From https://github.com/fudan-zvg/SETR/blob/c96656ee3ba7a4d70849d9e7ac1284759cf9d8f6/mmseg/models/decode_heads/vit_up_head.py
class DecoderDoubleUpsampling(nn.Module):
    """
    Two 1x1 convolutions. Each convolution is followed by a upsampling through interpolation.
    """
    def __init__(self, d_encoder, embedding_size, hidden_layer_size=256, n_cls=2):
        super().__init__()

        self.conv_0 = nn.Conv2d(d_encoder, hidden_layer_size, 1, 1)
        self.conv_1 = nn.Conv2d(hidden_layer_size, n_cls, 1, 1)
        self.norm = nn.LayerNorm((hidden_layer_size, embedding_size, embedding_size))

    def forward(self, x):
        x = self.conv_0(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*4, mode='bilinear', align_corners=False
        )
        x = self.conv_1(x)
        x = F.interpolate(
            x, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False
        )
        return x


class SatMaeSegmenterWithDoubleUpsampling(SatMaePretrained):
    def __init__(self, vit_size):
        super().__init__(vit_size)
        self.decoder = DecoderDoubleUpsampling(
            d_encoder=self.encoder_real_depth,
            embedding_size=self.n_patches_along_axis # h = w of the encoded embedding that the decoder receives
        )

    def forward(self, im):
        x = self.encoder(im)
        
        # remove CLS/DIST tokens for decoding
        x = x[:, 1:]  # (N, gL, D)

        # g is the number of channel groups
        x = rearrange(x, "N (g L) D -> N L (g D)", g=self.n_channel_groups)

        # In the following, h = H/P and w = W/P
        # where H is the height in the input image and P is the height and widt of each patch
        x = rearrange(x, "N (h w) cD -> N cD h w", h=self.n_patches_along_axis)

        upsampled_logits = self.decoder(x)
        return upsampled_logits
