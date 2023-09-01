# Adaption of SegformerForSemanticSegmentation to make it 
# integrate with Rastervision's SemanticSegmentationLearner

import torch
from torch import nn

from transformers import SegformerForSemanticSegmentation


class SegformerForSemanticSegmentationForRV(SegformerForSemanticSegmentation):
    """
    Changes compared to the original SegformerForSemanticSegmentation:
    - Output is simplified. The forward() funtion only output a Tensor with the logit values
    - The output tensor has the same width and height as the input image. We upsample by interpolation.
    
    We do this because Rastervision's SemanticSegmentationLearner expects the output in this formar.
    """
    def __init__(self, config, img_size):
        self.img_size = img_size # Tuple (width x height)
        super().__init__(config)
    
    def forward(self, pixel_values: torch.FloatTensor):
        outputs = self.segformer(
            pixel_values,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        )
        encoder_hidden_states = outputs.hidden_states
        logits = self.decode_head(encoder_hidden_states)
        upsampled_logits = nn.functional.interpolate(
            logits, size=self.img_size, mode="bilinear", align_corners=False
        )
        return upsampled_logits
