from enum import Enum
import numpy as np
from typing import Optional

from rastervision.core.data.raster_transformer.raster_transformer import (
    RasterTransformer,
)

# BSP_S1_MEAN and BSP_S1_STD are derived from our own dataset
BSP_S1_MEAN = np.array([-11.950798465937643, -18.939975061395252])  # [VV, VH]
BSP_S1_STD = np.array([3.319216134000598, 3.840950717746793])  # [VV, VH]


# SATMEA_S2_MEAN and SATMEA_S2_STD are copied from SatMAE
SATMEA_S2_MEAN = np.array(
    [
        1370.19151926,
        1184.3824625,
        1120.77120066,
        1136.26026392,
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        582.72633433,
        # 14.77112979,  # B10 commenting out because we it's not in our .tif EE export
        1732.16362238,
        1247.91870117,
    ]
)
SATMEA_S2_STD = np.array(
    [
        633.15169573,
        650.2842772,
        712.12507725,
        965.23119807,
        948.9819932,
        1108.06650639,
        1258.36394548,
        1233.1492281,
        1364.38688993,
        472.37967789,
        # 14.3114637 ,  # B10 commenting out because we it's not in our .tif EE export
        1310.36996126,
        1087.6020813,
    ]
)


class NormTransformer(RasterTransformer):
    """
    This transformer norms input data into the range 0 to 255.
    For each channel indivually:
    - Every value below mean - 2*std is clipped to 255
    - Every value above mean + 2*std is clipped to 255
    - All values in between are mapped with the range 0 to 255
    """

    def __init__(self, mean: np.array, std: np.array):
        # mean and std are statistical values per channel (!)
        self.min_value = mean - 2 * std
        self.max_value = mean + 2 * std

    def transform(
        self, chip: np.ndarray, channel_order: Optional[list] = None
    ) -> np.ndarray:
        min_value_filtered_channels = self.min_value[channel_order]
        max_value_filtered_channels = self.max_value[channel_order]
        chip = (
            (chip - min_value_filtered_channels)
            / (max_value_filtered_channels - min_value_filtered_channels)
            * 255.0
        )
        chip = np.clip(chip, 0, 255).astype(np.uint8)
        return chip


norm_s2_transformer = NormTransformer(mean=SATMEA_S2_MEAN, std=SATMEA_S2_STD)
norm_s1_transformer = NormTransformer(mean=BSP_S1_MEAN, std=BSP_S1_STD)


class DivideByConstantTransformer(RasterTransformer):
    """
    Divide all bands by a constant value and then clip between 0 and 1
    """

    def __init__(self, constant, min=0.0, max=1.0):
        self.min = min
        self.max = max
        self.constant = constant

    def transform(
        self, chip: np.ndarray, channel_order: Optional[list] = None
    ) -> np.ndarray:
        return np.clip(chip / self.constant, self.min, self.max)


divide_by_10000_transformer = DivideByConstantTransformer(10000)
divide_by_32_satlas_transformer = DivideByConstantTransformer(
    constant=32, min=0.0, max=255.0
)
