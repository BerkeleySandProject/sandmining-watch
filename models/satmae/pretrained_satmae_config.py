CHANNEL_GROUPS = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
PATCH_SIZE = 8
INPUT_SIZE = 96

from project_config import S2Band

satmea_pretrained_encoder_bands = [
    # dropping B1
    S2Band.B2,
    S2Band.B3,
    S2Band.B4,
    S2Band.B5,
    S2Band.B6,
    S2Band.B7,
    S2Band.B8,
    S2Band.B8A,
    # dropping B9
    # dropping B10
    S2Band.B11,
    S2Band.B12,
]
satmea_pretrained_encoder_bands_idx: list[int] = [
    e.value for e in satmea_pretrained_encoder_bands
]
