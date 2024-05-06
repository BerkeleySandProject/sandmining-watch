# Mostly copied from https://github.com/allenai/satlaspretrain_models/blob/main/satlaspretrain_models/utils.py
from enum import Enum, auto


class Backbone(Enum):
    SWINB = auto()
    SWINT = auto()
    RESNET50 = auto()
    RESNET152 = auto()


class Head(Enum):
    CLASSIFY = auto()
    MULTICLASSIFY = auto()
    DETECT = auto()
    INSTANCE = auto()
    SEGMENT = auto()
    BINSEGMENT = auto()
    REGRESS = auto()


# Dictionary of arguments needed to load in each SatlasPretrain pretrained model.
SatlasPretrain_weights = {
    # "Sentinel2_SwinB_SI_RGB": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_rgb.pth?download=true",
    #     "backbone": Backbone.SWINB,
    #     "num_channels": 3,
    #     "multi_image": False,
    # },
    # "Sentinel2_SwinB_MI_RGB": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_mi_rgb.pth?download=true",
    #     "backbone": Backbone.SWINB,
    #     "num_channels": 3,
    #     "multi_image": True,
    # },
    "Sentinel2_SwinB_SI_MS": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_ms.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 9,
        "multi_image": False,
    },
    # "Sentinel2_SwinB_MI_MS": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_mi_ms.pth?download=true",
    #     "backbone": Backbone.SWINB,
    #     "num_channels": 9,
    #     "multi_image": True,
    # },
    "Sentinel1_SwinB_SI": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel1_swinb_si.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 2,
        "multi_image": False,
    },
    # "Sentinel1_SwinB_MI": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel1_swinb_mi.pth?download=true",
    #     "backbone": Backbone.SWINB,
    #     "num_channels": 2,
    #     "multi_image": True,
    # },
    # "Landsat_SwinB_SI": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/landsat_swinb_si.pth?download=true",
    #     "backbone": Backbone.SWINB,
    #     "num_channels": 11,
    #     "multi_image": False,
    # },
    # "Landsat_SwinB_MI": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/landsat_swinb_mi.pth?download=true",
    #     "backbone": Backbone.SWINB,
    #     "num_channels": 11,
    #     "multi_image": True,
    # },
    # "Aerial_SwinB_SI": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_si.pth?download=true",
    #     "backbone": Backbone.SWINB,
    #     "num_channels": 3,
    #     "multi_image": False,
    # },
    # "Aerial_SwinB_MI": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_mi.pth?download=true",
    #     "backbone": Backbone.SWINB,
    #     "num_channels": 3,
    #     "multi_image": True,
    # },
    # "Sentinel2_SwinT_SI_RGB": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_rgb.pth?download=true",
    #     "backbone": Backbone.SWINT,
    #     "num_channels": 3,
    #     "multi_image": False,
    # },
    # "Sentinel2_SwinT_SI_MS": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_ms.pth?download=true",
    #     "backbone": Backbone.SWINT,
    #     "num_channels": 9,
    #     "multi_image": False,
    # },
    # "Sentinel2_SwinT_MI_RGB": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_mi_rgb.pth?download=true",
    #     "backbone": Backbone.SWINT,
    #     "num_channels": 3,
    #     "multi_image": True,
    # },
    # "Sentinel2_SwinT_MI_MS": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_mi_ms.pth?download=true",
    #     "backbone": Backbone.SWINT,
    #     "num_channels": 9,
    #     "multi_image": True,
    # },
    # "Sentinel2_Resnet50_SI_RGB": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_si_rgb.pth?download=true",
    #     "backbone": Backbone.RESNET50,
    #     "num_channels": 3,
    #     "multi_image": False,
    # },
    # "Sentinel2_Resnet50_SI_MS": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_si_ms.pth?download=true",
    #     "backbone": Backbone.RESNET50,
    #     "num_channels": 9,
    #     "multi_image": False,
    # },
    # "Sentinel2_Resnet50_MI_RGB": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_mi_rgb.pth?download=true",
    #     "backbone": Backbone.RESNET50,
    #     "num_channels": 3,
    #     "multi_image": True,
    # },
    # "Sentinel2_Resnet50_MI_MS": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_mi_ms.pth?download=true",
    #     "backbone": Backbone.RESNET50,
    #     "num_channels": 9,
    #     "multi_image": True,
    # },
    # "Sentinel2_Resnet152_SI_RGB": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_si_rgb.pth?download=true",
    #     "backbone": Backbone.RESNET152,
    #     "num_channels": 3,
    #     "multi_image": False,
    # },
    "Sentinel2_Resnet152_SI_MS": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_si_ms.pth?download=true",
        "backbone": Backbone.RESNET152,
        "num_channels": 9,
        "multi_image": False,
    },
    # "Sentinel2_Resnet152_MI_RGB": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_mi_rgb.pth?download=true",
    #     "backbone": Backbone.RESNET152,
    #     "num_channels": 3,
    #     "multi_image": True,
    # },
    # "Sentinel2_Resnet152_MI_MS": {
    #     "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_mi_ms.pth?download=true",
    #     "backbone": Backbone.RESNET152,
    #     "num_channels": 9,
    #     "multi_image": True,
    # },
}


def adjust_state_dict_prefix(
    state_dict, needed, prefix=None, prefix_allowed_count=None
):
    """
    Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with 'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component.
        if not needed in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, "", 1)

        new_state_dict[key] = value
    return new_state_dict


class Weights:
    def __init__(self):
        """
        Class to manage downloading weights formatting them to be loaded into SatlasPretrain models, and caching them on our servers.
        """
        super(Weights, self).__init__()

    def get_pretrained_model(
        self, model_identifier, fpn=False, head=None, num_categories=None, device="cuda"
    ):
        """
        Find and load pretrained SatlasPretrain weights, based on the model_identifier argument.
        Option to load pretrained FPN and/or a randomly initialized head.

        Args:
            model_identifier:
            fpn (bool): Whether or not to load a pretrained FPN along with the Backbone.
            head (enum): If specified, a randomly initialized Head will be created along with the
                        Backbone and [optionally] the FPN.
            num_categories (int): Number of categories to be included in output from prediction head.
        """
        # Validate that the model identifier is supported.
        if not model_identifier in SatlasPretrain_weights.keys():
            raise ValueError(
                "Invalid model_identifier. See utils.SatlasPretrain_weights."
            )

        if head and (num_categories is None):
            raise ValueError("Must specify num_categories if head is desired.")

        model_info = SatlasPretrain_weights[model_identifier]

        # Use hardcoded huggingface url to download weights.
        weights_url = model_info["url"]
        response = requests.get(weights_url)
        if response.status_code == 200:
            weights_file = BytesIO(response.content)
        else:
            raise Exception(f"Failed to download weights from {url}")

        if device == "cpu":
            weights = torch.load(
                weights_file, map_location=torch.device("cpu"))
        else:
            weights = torch.load(weights_file)

        # Initialize a pretrained model using the Model() class.
        model = SatlasPretrain(
            model_info["num_channels"],
            model_info["multi_image"],
            model_info["backbone"],
            fpn=fpn,
            head=head,
            num_categories=num_categories,
            weights=weights,
        )
        return model
