from project_config import S2Band

CHANNEL_GROUPS = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
# Group 0: B1, B2, B3, B7
# Group 1: B4, B5, B6, B8
# Group 2: B8A, B9
PATCH_SIZE = 8
INPUT_SIZE = 96

# https://github.com/allenai/satlas/blob/main/Normalization.md
# Multispectral normalization documentation
satlas_pretrained_encoder_bands = [
    # dropping B1
    S2Band.B2,
    S2Band.B3,
    S2Band.B4,
    S2Band.B5,
    S2Band.B6,
    S2Band.B7,
    S2Band.B8,
    # dropping B8a
    # dropping B9
    # dropping B10
    S2Band.B11,
    S2Band.B12,
]
satlas_pretrained_encoder_bands_idx: list[int] = [
    e.value for e in satlas_pretrained_encoder_bands
]
