import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..satmae_pretrained_encoder import SatMaePretrained


# From https://github.com/fudan-zvg/SETR/blob/c96656ee3ba7a4d70849d9e7ac1284759cf9d8f6/mmseg/models/decode_heads/vit_up_head.py
class DecoderDoubleUpsampling(nn.Module):
    """
    Two 1x1 convolutions. Each convolution is followed by a upsampling through interpolation.
    """
    def __init__(self, d_encoder, embedding_size, image_size, num_classes, hidden_layer_size=256):
        super().__init__()
        self.image_size = image_size
        self.conv_0 = nn.Conv2d(d_encoder, hidden_layer_size, 1, 1)
        self.conv_1 = nn.Conv2d(hidden_layer_size, num_classes, 1, 1)
        self.norm = nn.LayerNorm((hidden_layer_size, embedding_size, embedding_size)) # does not scale with changing input size
        # self.norm = nn.BatchNorm2d(hidden_layer_size)
        #self.norm = nn.LayerNorm(hidden_layer_size)  # Normalize across channel dimension only -> this allows us to change he width and height

    def forward(self, x):
        x = self.conv_0(x)
        x = self.norm(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*4, mode='bilinear', align_corners=False
        )
        x = self.conv_1(x)
        x = F.interpolate(
            x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
        )
        return x
    
class DecoderTripleUpsampling3x3(nn.Module):
    """
    Three 3x3 convolutions. Each convolution is followed by a upsampling through interpolation.
    """
    def __init__(self, d_encoder, embedding_size, image_size, num_classes, hidden_layer_size=256):
        super().__init__()
        self.image_size = image_size
        self.conv_0 = nn.Conv2d(d_encoder, hidden_layer_size, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(hidden_layer_size, hidden_layer_size, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(hidden_layer_size, num_classes, kernel_size=1, stride=1)
        self.norm_0 = nn.BatchNorm2d(hidden_layer_size)
        self.norm_1 = nn.BatchNorm2d(hidden_layer_size)
        # self.norm = nn.BatchNorm2d(hidden_layer_size)
        #self.norm = nn.LayerNorm(hidden_layer_size)  # Normalize across channel dimension only -> this allows us to change he width and height

    def forward(self, x):
        x = self.conv_0(x)
        x = self.norm_0(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(
            x, size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        
        x = self.conv_2(x)
        x = F.interpolate(
            x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False) #in theory this is another 2x upsampling
        return x
    



# class DecoderDoubleUpsampling(nn.Module):
#     """
#     Two 1x1 convolutions. Each convolution is followed by a upsampling through interpolation.
#     """

#     def __init__(self, d_encoder, embedding_size, image_size, num_classes, hidden_layer_size=256):
#         super().__init__()
#         self.image_size = image_size
#         self.conv_0 = nn.Conv2d(d_encoder, hidden_layer_size, 1, 1)
#         self.deconv_0 = nn.ConvTranspose2d(hidden_layer_size, hidden_layer_size, kernel_size=4, stride=4, padding=0)
#         self.conv_1 = nn.Conv2d(hidden_layer_size, num_classes, 1, 1)
#         self.deconv_1 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=4, padding=0)
#         self.norm = nn.BatchNorm2d(hidden_layer_size)

#     def forward(self, x):
#         x = F.relu(self.conv_0(x), inplace=True)
#         x = F.relu(self.deconv_0(x), inplace=True)
#         x = F.relu(self.conv_1(x), inplace=True)
#         x = F.relu(self.deconv_1(x), inplace=True)
#         return x
    




class SatMaeSegmenterWithDoubleUpsampling(SatMaePretrained):
    def __init__(self, vit_size, image_size, num_classes, num_levels=2):
        super().__init__(vit_size, image_size)

        if num_levels is None or num_levels == 2:

            self.decoder = DecoderDoubleUpsampling(
                d_encoder=self.encoder_real_depth,
                embedding_size=self.n_patches_along_axis, # h = w of the encoded embedding that the decoder receives
                image_size=image_size,
                num_classes=num_classes,
            )

        elif num_levels == 3:
                
                self.decoder = DecoderTripleUpsampling3x3(
                    d_encoder=self.encoder_real_depth,
                    embedding_size=self.n_patches_along_axis, # h = w of the encoded embedding that the decoder receives
                    image_size=image_size,
                    num_classes=num_classes,
                )

    def forward(self, im):
        x = self.encoder(im)
        
        # remove CLS/DIST tokens for decoding
        x = x[:, 1:]  # (N, gL, D)

        # g is the number of channel groups
        x = rearrange(x, "N (g L) D -> N L (g D)", g=self.n_channel_groups)

        # In the following, h = H/P and w = W/P
        # where H is the height in the input image and P is the height and width of each patch
        x = rearrange(x, "N (h w) cD -> N cD h w", h=self.n_patches_along_axis)

        upsampled_logits = self.decoder(x)
        return upsampled_logits
    

# from torch.nn import TransformerEncoder, TransformerEncoderLayer
# import torch.nn as nn

# class TransformerDecoderHead(nn.Module):
#     def __init__(self, in_channels, num_classes, image_size):
#         super().__init__()

#         encoder_layers = TransformerEncoderLayer(d_model=in_channels, nhead=8)
#         self.transformer = TransformerEncoder(encoder_layers, num_layers=2)
#         # Add transposed convolution layers
#         self.deconv1 = nn.ConvTranspose2d(in_channels, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv2 = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.deconv4 = nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)



#         self.image_size = image_size

#     def forward(self, x):
#         # Flatten the spatial dimensions of the feature map
#         x = x.view(x.size(0), x.size(1), -1).permute(2, 0, 1)

#         # print(x.shape)

#         # Apply the transformer
#         x = self.transformer(x)

#         # print(x.shape)

#         # Reshape the output to the original image size
#         x = x.permute(1, 2, 0).view(x.size(1), -1, 20, 20)
#         # print("Final reshape before conv ", x.shape)
        

#         # Apply the transposed convolution layers
#         x = F.relu(self.deconv1(x))
#         # print(x.shape)
#         x = F.relu(self.deconv2(x))
#         # print(x.shape)
#         x = F.relu(self.deconv3(x))
#         # print(x.shape)
#         x = self.deconv4(x)
#         # print(x.shape)
#         #add an interpolation layer to output to the original image size
#         x = F.interpolate(
#             x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
#         )
#         # print('After interpolation ', x.shape)

#         return x



# class SatMaeSegmenterWithViTDecoder(SatMaePretrained):
#     def __init__(self, vit_size, image_size, num_classes):
#         super().__init__(vit_size, image_size)
#         self.decoder = TransformerDecoderHead(
#             in_channels=self.encoder_real_depth,
#             image_size=image_size,
#             num_classes=num_classes,
#         )
#         # print("ViT decoder initialized ", self.encoder_real_depth)
        

#     def forward(self, im):
#         x = self.encoder(im)

#         # print(im.shape, x.shape)
        
#         # remove CLS/DIST tokens for decoding
#         x = x[:, 1:]  # (N, gL, D)

#         # g is the number of channel groups
#         x = rearrange(x, "N (g L) D -> N L (g D)", g=self.n_channel_groups)

#         # print(x.shape)

#         # # In the following, h = H/P and w = W/P
#         # # where H is the height in the input image and P is the height and widt of each patch
#         x = rearrange(x, "N (h w) cD -> N cD h w", h=self.n_patches_along_axis)

#         # print(x.shape)

#         upsampled_logits = self.decoder(x)
#         return upsampled_logits
