import os
import torch
from torch import nn

from .models_vit_group_channels import vit_base_patch16, vit_large_patch16
from .util.pos_embed import interpolate_pos_embed
from .pretrained_satmae_config import CHANNEL_GROUPS, PATCH_SIZE

class SatMaePretrained(nn.Module):
    """
    A SatMaePretrained holds the pretrained SatMAE spectral encoder in the 'base' size.
    We inherited from this class to construct SatMAE encoder + custom decoder models.
    """
    def __init__(self, vit_size, image_size):
        super().__init__()
        if vit_size == "base":
            encoder_factory_fcn = vit_base_patch16
        elif vit_size == "large":
            encoder_factory_fcn = vit_large_patch16
        self.encoder = encoder_factory_fcn(
            patch_size=PATCH_SIZE, img_size=image_size,
            channel_groups=CHANNEL_GROUPS,
            num_classes=2, drop_path_rate=0.1, global_pool=False,
            use_encoder_only=True,
        )
        self.n_channel_groups = len(CHANNEL_GROUPS)
        self.encoder_real_depth = self.encoder.embed_dim * self.n_channel_groups
        self.n_patches_along_axis = image_size // PATCH_SIZE # In SatMAE notation: H/P or W/P (because H=W, input is quadratic)

    def load_encoder_weights(self, path_to_weights):
        if not os.path.isfile(path_to_weights):
            raise ValueError(f"No checkpoint found at {path_to_weights}")
        print(f"SatMae: Loading encoder weights from {path_to_weights}")
        checkpoint = torch.load(path_to_weights, map_location='cpu')

        #check if checkpoint has a 'model' key
        # this indicates that you're loading the original SatMAE pretrained weights
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
            interpolate_pos_embed(self.encoder, checkpoint_model)
            msg = self.encoder.load_state_dict(checkpoint_model, strict=False)


        else: #you're loading your own checkpoints which have the encoder and decoder weights separately saved as keys
            checkpoint_model = checkpoint
            state_dict = self.encoder.state_dict()

            #rename the encoder weights -> get rid of 'encoder.' prefix
            checkpoint_encoder = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if 'encoder' in k}

            ## These are parameters that are not learnt and can be removed -> need this when loading the original SatMAE weights
            # for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
            #     if k in checkpoint_encoder and checkpoint_encoder[k].shape != state_dict[k].shape:
            #         print(f"Removing key {k} from pretrained checkpoint")
            #         del checkpoint_encoder[k]            
 
            interpolate_pos_embed(self.encoder, checkpoint_encoder)

            #rename the encoder weights
            # checkpoint_encoder = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if 'encoder' in k}
            
            msg = self.encoder.load_state_dict(checkpoint_encoder, strict=False)  #since the model params cons

        if msg.missing_keys:
            print("Warning! Missing keys:")
            print(msg.missing_keys)
        # print(msg) 
        
    def freeze_encoder_weights(self):
        print("SatMaePretrained: Freezing encoder weights")
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_embed_weights(self):
        print("SatMaePretrained: Freezing weights of embedding layer")
        for param in self.encoder.patch_embed.parameters():
            param.requires_grad = False

    def load_decoder_weights(self, path_to_weights):
        """
        Loads the decoder weights from a checkpoint
        """
        if not os.path.isfile(path_to_weights):
            raise ValueError(f"No checkpoint found at {path_to_weights}")
        print(f"SatMae: Loading decoder weights from {path_to_weights}")
        checkpoint = torch.load(path_to_weights, map_location='cpu')
        checkpoint_decoder = {k.replace('decoder.', ''): v for k, v in checkpoint.items() if 'decoder' in k}
        
        state_dict = self.decoder.state_dict()

        self.decoder.load_state_dict(checkpoint_decoder, strict=False)
