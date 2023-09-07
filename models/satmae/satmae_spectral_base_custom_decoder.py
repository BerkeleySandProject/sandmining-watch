import torch
from torch import nn
from einops import rearrange

from .models_vit_group_channels import vit_base_patch16
from .decoders.decoder_linear import DecoderLinear
from .util.pos_embed import interpolate_pos_embed
from .pretrained_satmae_config import CHANNEL_GROUPS, PATCH_SIZE, INPUT_SIZE


class SatMaeSegmenterWithLinearDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = vit_base_patch16(
            patch_size=PATCH_SIZE, img_size=INPUT_SIZE, in_chans=10,
            channel_groups=CHANNEL_GROUPS,
            num_classes=2, drop_path_rate=0.1, global_pool=False,
            use_encoder_only=True,
        )
        self.n_channel_groups = len(CHANNEL_GROUPS)

        self.decoder = DecoderLinear(
            n_cls=2,
            patch_size=PATCH_SIZE,
            d_encoder=self.encoder.embed_dim * self.n_channel_groups,
        )

    def load_encoder_weights(self, path_to_weights):
        print(f"SatMaeSegmenterWithLinearDecoder: Loading encoder weights from {path_to_weights}")
        checkpoint = torch.load(path_to_weights, map_location='cpu')
        checkpoint_model = checkpoint['model']

        state_dict = self.encoder.state_dict()
        for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(self.encoder, checkpoint_model)
        msg = self.encoder.load_state_dict(checkpoint_model, strict=False)
        print(msg) 
        
    def freeze_encoder_weights(self):
        print("SatMaeSegmenterWithLinearDecoder: Freezing encoder weights")
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Except channel_cls_embed. These weights are not loaded from checkpoint. So I think we need to learn them.
        self.encoder.channel_cls_embed.requires_grad = True

    def forward(self, im):
        x = self.encoder(im)
        
        # remove CLS/DIST tokens for decoding
        x = x[:, 1:]  # (N, cL, D)

        x = rearrange(x, 'N (c L) D -> N L (c D)', c=self.n_channel_groups)

        logits = self.decoder(x, (INPUT_SIZE, INPUT_SIZE))
        upsampled_logits = nn.functional.interpolate(
            logits, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
        )
        return upsampled_logits
