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
