import numpy as np
from math import ceil
from rastervision.core.data import (
    ClassInferenceTransformer, GeoJSONVectorSource,
    RasterioSource, MultiRasterSource, RasterizedSource, Scene,
    SemanticSegmentationLabelSource
)
from rastervision.core.data.raster_transformer.nan_transformer import NanTransformer

from rastervision.core.data.raster_source import RasterSource
from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset, SemanticSegmentationRandomWindowGeoDataset


from project_config import CLASS_NAME, CLASS_CONFIG, USE_RIVER_AOIS
from experiment_configs.schemas import SupervisedTrainingConfig, DatasetChoice, NormalizationS2Choice
from utils.schemas import ObservationPointer
from ml.norm_data import norm_s1_transformer, norm_s2_transformer, divide_by_10000_transformer
from ml.augmentations import DEFAULT_AUGMENTATIONS
from typing import TYPE_CHECKING, Optional, List
from shapely.geometry import Polygon

def observation_to_scene(config: SupervisedTrainingConfig, observation: ObservationPointer) -> Scene:
    if config.datasets == DatasetChoice.S1S2:
        return create_scene_s1s2(
            config,
            s2_uri=observation.uri_to_s2,
            s1_uri=observation.uri_to_s1,
            label_uri=observation.uri_to_annotations,
            scene_id=observation.name,
            rivers_uri=observation.uri_to_rivers
        )
    elif config.datasets == DatasetChoice.S2:
        return create_scene_s2(
            config,
            s2_uri=observation.uri_to_s2,
            label_uri=observation.uri_to_annotations,
            scene_id=observation.name,
            rivers_uri=observation.uri_to_rivers
        )
    elif config.datasets == DatasetChoice.S2_L1C:
        return create_scene_s2(
            config,
            s2_uri=observation.uri_to_s2_l1c,
            label_uri=observation.uri_to_annotations,
            scene_id=observation.name,
            rivers_uri=observation.uri_to_rivers
        )
    else:
        raise ValueError("Unexped value for config.datasets")

def create_scene_s1s2(config, s2_uri, s1_uri, label_uri, scene_id, rivers_uri) -> Scene:
    s1s2_source = create_s1s2_multirastersource(config, s2_uri, s1_uri)
    scene = rastersource_with_labeluri_to_scene(
        s1s2_source,
        label_uri,
        scene_id,
        rivers_uri
    )
    return scene

def create_scene_s2(config, s2_uri, label_uri, scene_id, rivers_uri) -> Scene:
    s2_source = create_s2_image_source(config, s2_uri)
    scene = rastersource_with_labeluri_to_scene(
        s2_source,
        label_uri,
        scene_id,
        rivers_uri
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

def rastersource_with_labeluri_to_scene(img_raster_source: RasterSource, label_uri, scene_id, rivers_uri) -> Scene:
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
        # extent=img_raster_source.extent
        bbox = img_raster_source.bbox
    )

    label_source = SemanticSegmentationLabelSource(
        label_raster_source, class_config=CLASS_CONFIG
    )

    if rivers_uri is not None and USE_RIVER_AOIS: #create the aoi_polygons 
        river_vector_source = GeoJSONVectorSource(
                        rivers_uri,
                        crs_transformer=img_raster_source.crs_transformer,
                        ignore_crs_field=True
                        )

        aoi_polygons = river_vector_source.get_geoms()
        scene = Scene(id=scene_id, raster_source=img_raster_source, label_source=label_source, aoi_polygons=aoi_polygons)    
    
    else:
        scene = Scene(id=scene_id, raster_source=img_raster_source, label_source=label_source)

    return scene

def sliding_filter_by_aoi(windows: List['Box'],
                    aoi_polygons: List[Polygon],
                    within: bool = True) -> List['Box']:
    """Filters windows by a list of AOI polygons

    Args:
        within: if True, windows are only kept if their centroid lies fully within an
            AOI polygon. Otherwise, windows are kept if they intersect an AOI
            polygon.
    """
    result = []
    for window in windows:
        w = window.to_shapely()
        for polygon in aoi_polygons:
            if (within and w.centroid.within(polygon)
                    or ((not within) and w.intersects(polygon))):
                result.append(window)
                break

    return result

def custom_init_windows(self) -> None:
    """Pre-compute windows."""

    windows = self.scene.raster_source.extent.get_windows(
        self.size,
        stride=self.stride,
        padding=self.padding,
        pad_direction=self.pad_direction)
    if len(self.scene.aoi_polygons) > 0:
        windows = sliding_filter_by_aoi(windows, self.scene.aoi_polygons , within=False)
        # windows = Box.filter_by_aoi(windows, self.scene.aoi_polygons, within=False)
    self.windows = windows

def scene_to_validation_ds(config, scene: Scene):
    # No augementation and windows don't overlap. Use for validation during training time.
    ds = SemanticSegmentationSlidingWindowGeoDataset(
        scene,
        size=config.tile_size,
        stride=config.tile_size,
        padding=config.tile_size,
        pad_direction='end',
        transform=None,
        normalize=True,
    )

    #override windows initialization with custom function
    ds.init_windows = custom_init_windows.__get__(ds, SemanticSegmentationSlidingWindowGeoDataset)
    #needs to be called again because it's only called during __init__, during which time the custom function has not been overridden
    ds.init_windows()

    return ds

    
from rastervision.core.box import Box
from typing import List
from shapely.geometry import Polygon

def custom_within_aoi(window: 'Box', aoi_polygons: List[Polygon], aoi_centroids) -> bool:
    """Check if window is within a list of AOI polygons."""

    w = window.to_shapely()
    for polygon in aoi_polygons:
        if aoi_centroids:
            if w.centroid.within(polygon):
                return True
        else:
            if w.within(polygon):
                return True
    return False

def custom_sample_window(self) -> Box:
        """If scene has AOI polygons, try to find a random window that is
        within the AOI. Otherwise, just return the first sampled window.

        Raises:
            StopIteration: If unable to find a valid window within
                self.max_sample_attempts attempts.

        Returns:
            Box: The sampled window.
        """

        if not self.has_aoi_polygons:
            window = self._sample_window()
            return window

        for _ in range(self.max_sample_attempts):
            window = self._sample_window()
            if custom_within_aoi(window, self.aoi_polygons, self.aoi_centroids):
                return window
        raise StopIteration('Failed to find random window within scene AOI.')

def scene_to_training_ds(config: SupervisedTrainingConfig, scene: Scene, aoi_centroids = True):
    
    """
    Returns a dataset for training. The dataset will sample windows from the scene in a random fashion.
    aoi_centroids: If True, the centroids of the AOI polygons are used to determine if a window is within the AOI,
                    otherwise the entire window needs to fit inside the AOI
    """

    aoi_area = 0
    for aoi in scene.aoi_polygons:
        aoi_area += aoi.area

    n_windows = int(np.ceil(aoi_area / config.tile_size ** 2)) * 2    
    # n_windows = ceil(n_pixels_in_scene / config.tile_size ** 2)
    ds = SemanticSegmentationRandomWindowGeoDataset(
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

    #override the sample_window method and within_aoi method
    ds.aoi_centroids = aoi_centroids
    ds.sample_window = custom_sample_window.__get__(ds, SemanticSegmentationRandomWindowGeoDataset)
    return ds
