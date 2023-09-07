import numpy as np
from typing import Optional

from rastervision.core.data.raster_transformer.raster_transformer import RasterTransformer


class NormS2Transformer(RasterTransformer):
    NORM_FACTOR = 10000
    def transform(self, chip: np.ndarray,
                  channel_order: Optional[list] = None) -> np.ndarray:
        return chip / self.NORM_FACTOR

class NormS1Transformer(RasterTransformer):
    def transform(self, chip: np.ndarray,
                  channel_order: Optional[list] = None) -> np.ndarray:
        return chip / 30 + 1

SATMEA_NORM_MEAN = np.array([
            1370.19151926, 
            1184.3824625 , 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,  582.72633433,
            # 14.77112979,  # B10 commenting out because we it's not in our .tif EE export
            1732.16362238, 1247.91870117])
SATMEA_NORM_STD = np.array([
            633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
            948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281,
            1364.38688993,  472.37967789, 
            #14.3114637 ,  # B10 commenting out because we it's not in our .tif EE export
            1310.36996126, 1087.6020813])

class SatMaeNormS2Transformer(RasterTransformer):
    """
    The SatMae encoder was learned using this.
    """
    def __init__(self):
        self.min_value = SATMEA_NORM_MEAN - 2 * SATMEA_NORM_STD
        self.max_value = SATMEA_NORM_MEAN + 2 * SATMEA_NORM_STD
        
    def transform(self, chip: np.ndarray,
                  channel_order: Optional[list] = None) -> np.ndarray:
        min_value_filtered_channels = self.min_value[channel_order]
        max_value_filtered_channels = self.max_value[channel_order]
        chip = (chip - min_value_filtered_channels) / (max_value_filtered_channels - min_value_filtered_channels) * 255.0
        chip = np.clip(chip, 0, 255).astype(np.uint8)
        return chip
