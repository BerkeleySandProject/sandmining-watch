import torch
from torch import nn

from .models_vit_group_channels import vit_base_patch16
from .util.pos_embed import interpolate_pos_embed
from .pretrained_satmae_config import CHANNEL_GROUPS, PATCH_SIZE, INPUT_SIZE

class SatMaePretrained(nn.Module):
    """
    A SatMaePretrained holds the pretrained SatMAE spectral encoder in the 'base' size.
    We inherited from this class to construct SatMAE encoder + custom decoder models.
    """
    def __init__(self):
        super().__init__()
        self.encoder = vit_base_patch16(
            patch_size=PATCH_SIZE, img_size=INPUT_SIZE, in_chans=10,
            channel_groups=CHANNEL_GROUPS,
            num_classes=2, drop_path_rate=0.1, global_pool=False,
            use_encoder_only=True,
        )
        self.n_channel_groups = len(CHANNEL_GROUPS)
        self.encoder_real_depth = self.encoder.embed_dim * self.n_channel_groups
        self.n_patches_along_axis = INPUT_SIZE // PATCH_SIZE # In SatMAE notation: H/P or W/P (because H=W, input is quadratic)

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
        print("SatMaePretrained: Freezing encoder weights")
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Except channel_cls_embed. These weights are not loaded from checkpoint. We want to learn them.
        self.encoder.channel_cls_embed.requires_grad = True
