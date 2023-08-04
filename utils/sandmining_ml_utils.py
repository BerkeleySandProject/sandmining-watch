import numpy as np
from typing import Optional

from rastervision.core.data.raster_transformer.raster_transformer import RasterTransformer


class NormS2Transformer(RasterTransformer):
    NORM_FACTOR = 10000
    def transform(self, chip: np.ndarray,
                  channel_order: Optional[list] = None) -> np.ndarray:
        return chip / self.NORM_FACTOR

class NormS1Transformer(RasterTransformer):
    # TODO
    def transform(self, chip: np.ndarray,
                  channel_order: Optional[list] = None) -> np.ndarray:
        return chip

class RemoveNanTransformer(RasterTransformer):
    # Replaces nan values with zeroes.
    def transform(self, chip: np.ndarray, channel_order) -> np.ndarray:
        return np.nan_to_num(chip, copy=False, nan=0.0)
