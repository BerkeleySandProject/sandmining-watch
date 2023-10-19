from typing import List
import numpy as np
from math import ceil
from rastervision.core.box import Box
from rastervision.core.data import (
    ClassInferenceTransformer, GeoJSONVectorSource, RasterTransformer,
    RasterioSource, MultiRasterSource, RasterizedSource, Scene,
    SemanticSegmentationLabelSource, VectorSource
)
from rastervision.core.data.raster_transformer.nan_transformer import NanTransformer

from rastervision.core.data.raster_source import RasterSource
from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset, SemanticSegmentationRandomWindowGeoDataset


from project_config import CLASS_NAME, CLASS_CONFIG
from experiment_configs.schemas import SupervisedTrainingConfig, DatasetChoice, NormalizationS2Choice
from utils.schemas import ObservationPointer
from ml.norm_data import norm_s1_transformer, norm_s2_transformer, divide_by_10000_transformer
from ml.augmentations import DEFAULT_AUGMENTATIONS

from rasterio.features import rasterize, MergeAlg
import geopandas as gpd


def observation_to_scene(config: SupervisedTrainingConfig, observation: ObservationPointer) -> Scene:
    if config.datasets == DatasetChoice.S1S2:
        return create_scene_s1s2(
            config,
            s2_uri=observation.uri_to_s2,
            s1_uri=observation.uri_to_s1,
            label_uri=observation.uri_to_annotations,
            scene_id=observation.name,
        )
    elif config.datasets == DatasetChoice.S2:
        return create_scene_s2(
            config,
            s2_uri=observation.uri_to_s2,
            label_uri=observation.uri_to_annotations,
            scene_id=observation.name,
        )
    elif config.datasets == DatasetChoice.S2_L1C:
        return create_scene_s2(
            config,
            s2_uri=observation.uri_to_s2_l1c,
            label_uri=observation.uri_to_annotations,
            scene_id=observation.name,
        )
    else:
        raise ValueError("Unexped value for config.datasets")

def create_scene_s1s2(config, s2_uri, s1_uri, label_uri, scene_id) -> Scene:
    s1s2_source = create_s1s2_multirastersource(config, s2_uri, s1_uri)
    scene = rastersource_with_labeluri_to_scene(
        s1s2_source,
        label_uri,
        scene_id
    )
    return scene

def create_scene_s2(config, s2_uri, label_uri, scene_id) -> Scene:
    s2_source = create_s2_image_source(config, s2_uri)
    scene = rastersource_with_labeluri_to_scene(
        s2_source,
        label_uri,
        scene_id
    )
    return scene

def create_s2_image_source(config, img_uri):
    if config.s2_normalization == NormalizationS2Choice.ChannelWise:
        normalization_transformer = norm_s2_transformer
    elif config.s2_normalization == NormalizationS2Choice.DivideBy10000:
        normalization_transformer = divide_by_10000_transformer
    else:
        raise ValueError("Unsupported value for config.s2_normalization")
    
    return RasterioSource(
        img_uri,
        channel_order=config.s2_channels,
        allow_streaming=False,
        raster_transformers=[
            NanTransformer(),
            normalization_transformer,
        ]
    )

def create_s1_image_source(config, img_uri):
    return RasterioSource(
        img_uri,
        channel_order=None,
        allow_streaming=False,
        raster_transformers=[
            NanTransformer(),
            norm_s1_transformer
        ]
    )

def create_s1s2_multirastersource(config, s2_uri, s1_uri) -> MultiRasterSource:
    s2_source = create_s2_image_source(config, s2_uri)
    s1_source = create_s1_image_source(config, s1_uri)
    s1s2_source = MultiRasterSource(
        [s2_source, s1_source],
    )
    return s1s2_source

def warn_if_nan_in_raw_raster(raster_source):
    if isinstance(raster_source, MultiRasterSource):
        raster_sources = raster_source.raster_sources
    else:
        raster_sources = [raster_source]
    for raster_source in raster_sources:
        raw_image = raster_source.get_raw_image_array()
        if np.isnan(raw_image).any():
            print(f"WARNING: NaN in raw image {raster_source.uris}")

def rastersource_with_labeluri_to_scene(img_raster_source: RasterSource, label_uri, scene_id) -> Scene:
    vector_source = GeoJSONVectorSource(
        label_uri,
        img_raster_source.crs_transformer,
        ignore_crs_field=True,
        vector_transformers=[
            ClassInferenceTransformer(default_class_id=CLASS_CONFIG.get_class_id(CLASS_NAME))
        ]
    )

    label_raster_source = RasterizedSource(
        vector_source,
        background_class_id=CLASS_CONFIG.null_class_id,
        extent=img_raster_source.extent
    )

    label_source = SemanticSegmentationLabelSource(
        label_raster_source, class_config=CLASS_CONFIG
    )

    scene = Scene(id=scene_id, raster_source=img_raster_source, label_source=label_source)
    return scene

def scene_to_validation_ds(config, scene: Scene):
    # No augementation and windows don't overlap. Use for validation during training time.
    return SemanticSegmentationSlidingWindowGeoDataset(
        scene,
        size=config.tile_size,
        stride=config.tile_size,
        padding=config.tile_size,
        pad_direction='end',
        transform=None,
        normalize=True,
    )

def scene_to_training_ds(config: SupervisedTrainingConfig, scene: Scene):
    n_pixels_in_scene = scene.raster_source.shape[0] * scene.raster_source.shape[1]
    n_windows = ceil(n_pixels_in_scene / config.tile_size ** 2)
    return SemanticSegmentationRandomWindowGeoDataset(
        scene,
        out_size=(config.tile_size, config.tile_size),
        # Setting size_lims=(size,size+1) seems weird, but it actually leads to all windows having the same size
        # see https://github.com/azavea/raster-vision/blob/1d23e466d5bbec28373eef5c58efebcb0c774cd1/rastervision_pytorch_learner/rastervision/pytorch_learner/dataset/dataset.py#L408
        size_lims=(config.tile_size, config.tile_size+1),
        padding=None,
        max_windows=n_windows,
        efficient_aoi_sampling=True,
        transform=DEFAULT_AUGMENTATIONS,
        normalize=True,
    )

#Override RasterizedSource to use probability instead of integer class_ids
def merge_max(value):
    return np.max(value)

def geoms_to_raster_probability(df: gpd.GeoDataFrame, window: 'Box',
                    background_class_id: int, all_touched: bool,
                    extent: 'Box') -> np.ndarray:
    """Rasterize geometries that intersect with the window."""
    if len(df) == 0:
        return np.full(window.size, background_class_id, dtype=np.float32)

    window_geom = window.to_shapely()

    # subset to shapes that intersect window
    df_int = df[df.intersects(window_geom)]
    # transform to window frame of reference
    shapes = df_int.translate(xoff=-window.xmin, yoff=-window.ymin)
    # class IDs of each shape
    class_ids = df_int['class_id']

    print ("\n HERE! Class IDs: ", class_ids)

    # if len(shapes) > 0:
    #     raster = rasterize(
    #         shapes=list(zip(shapes, class_ids)),
    #         out_shape=window.size,
    #         fill=background_class_id,
    #         merge_alg=merge_max,#MergeAlg.max,  #for overlapping geometries, use the maximum value
    #         all_touched=all_touched)
    # else:
    #     raster = np.full(window.size, background_class_id, dtype=np.float32)

    #alternative way to rasterize, but it is slower
    # Rasterize each polygon separately
    rasters = []

    #iterate over each geometry in df_int, and rasterize it
    for i in range(len(df_int)):
        print(i,shapes.iloc[i], class_ids.iloc[i])
        raster = rasterize(
                    shapes = [(shapes.iloc[i], class_ids.iloc[i])],
                    out_shape = window.size,
                    fill = background_class_id,
                    all_touched=all_touched)
        print(raster)
        rasters.append(raster)  
    print('Done')
    #Now merge the rasters such that the maximum value is taken for each pixel
    raster = np.maximum.reduce(rasters)
    print('reduction done: ', raster)

    # for class_id in df['class_id'].unique():
    #     print('Processing ', class_id)
    #     mask = df['class_id'] == class_id
    #     shapes = list(zip(df.loc[mask, 'geometry'], np.ones(mask.sum()) * class_id))
    #     print(shapes)
    #     raster = rasterize(
    #         shapes=shapes,
    #         out_shape=window.size,
    #         fill=background_class_id,
    #         all_touched=all_touched
    #     )
    #     rasters.append(raster)

    # Take the element-wise maximum of the resulting rasters
    # raster = np.maximum.reduce(rasters)
    return raster


class RasterizedSourceProbability(RasterizedSource):
    def __init__(self, vector_source: VectorSource, 
                 background_class_id: int, 
                 extent: Box, all_touched: bool = False, 
                 raster_transformers: List[RasterTransformer] = ...):
        super().__init__(vector_source, background_class_id, extent, all_touched, raster_transformers)

    def _get_chip(self, window):
        """Return the chip located in the window.

        Polygons falling within the window are rasterized using the class_id, and
        the background is filled with background_class_id. Also, any pixels in the
        window outside the extent are zero, which is the don't-care class for
        segmentation.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        # log.debug(f'Rasterizing window: {window}')
        chip = geoms_to_raster_probability(
            self.df,
            window,
            background_class_id=self.background_class_id,
            extent=self.extent,
            all_touched=self.all_touched)
        # Add third singleton dim since rasters must have >=1 channel.
        chip = np.expand_dims(chip, 2)
        print('Chip shape: ', chip.shape)
        return chip
        return np.expand_dims(chip, 2)

