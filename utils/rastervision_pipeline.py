from typing import Any, List, Tuple, Optional, Union
import albumentations as A
import numpy as np
from math import ceil
from rastervision.core.box import Box
from rastervision.core.data import (
    ClassInferenceTransformer, GeoJSONVectorSource, RasterTransformer,
    RasterioSource, MultiRasterSource, RasterizedSource, Scene,
    SemanticSegmentationLabelSource, VectorSource
)
from rastervision.core.data.raster_transformer.nan_transformer import NanTransformer
from rastervision.core.data import LabelSource, LabelStore
from rastervision.core.data.raster_source import RasterSource

from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset, SemanticSegmentationRandomWindowGeoDataset
from rastervision.pytorch_learner import RandomWindowGeoDataset, SlidingWindowGeoDataset
from rastervision.pytorch_learner.dataset.transform import TransformType
from rastervision.pytorch_learner.learner_config import NonNegInt, PosInt

from shapely.geometry.base import BaseGeometry

import torch


from project_config import CLASS_NAME, CLASS_CONFIG, USE_RIVER_AOIS
from experiment_configs.schemas import SupervisedTrainingConfig, DatasetChoice, NormalizationS2Choice
from utils.schemas import ObservationPointer
from ml.norm_data import norm_s1_transformer, norm_s2_transformer, divide_by_10000_transformer
from ml.augmentations import DEFAULT_AUGMENTATIONS

from rasterio.features import rasterize, MergeAlg
import geopandas as gpd

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
    if label_uri is not None:
        vector_source = GeoJSONVectorSource(
            label_uri,
            img_raster_source.crs_transformer,
            ignore_crs_field=True,
            vector_transformers=[
                ClassInferenceTransformer(default_class_id=CLASS_CONFIG.get_class_id(CLASS_NAME))
            ]
        )
        label_raster_source = RasterizedSourceProbability(
            vector_source,
            background_class_id=CLASS_CONFIG.null_class_id,
            bbox=img_raster_source.bbox
        )
        label_source = SemanticSegmentationLabelSource(
            label_raster_source, class_config=CLASS_CONFIG
        )
        confidence_source = SemanticSegmentationLabelSource(
            RasterizedConfidences(
                vector_source,
                background_class_id=CLASS_CONFIG.null_class_id,
                bbox=img_raster_source.extent),
        class_config=CLASS_CONFIG
        )
    else:
        label_source = None

    if rivers_uri is not None and USE_RIVER_AOIS: #create the aoi_polygons 
        river_vector_source = GeoJSONVectorSource(
                        rivers_uri,
                        crs_transformer=img_raster_source.crs_transformer,
                        ignore_crs_field=True
                        )

        aoi_polygons = river_vector_source.get_geoms()
        scene = Scene(id=scene_id, raster_source=img_raster_source, label_source=label_source, aoi_polygons=aoi_polygons)    
    
    else:
        scene = SceneWithConfidences(id=scene_id, 
                        raster_source=img_raster_source, 
                        label_source=label_source, 
                        confidence_source=confidence_source)

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

def scene_to_validation_ds(config, scene: Scene, stride=None):
    if stride is None:
        stride = config.tile_size

    # No augementation and windows don't overlap. Use for validation during training time.
    ds = SemanticSegmentationSlidingWindowGeoDataset(
        scene,
        size=config.tile_size,
        stride=stride,
        padding=config.tile_size,
        pad_direction='end',
        transform=None,
        normalize=True, # If unsigned integer, bring to range [0, 1]
    )
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

    # n_windows = int(np.ceil(aoi_area / config.tile_size ** 2)) * 2    
    # n_windows = ceil(n_pixels_in_scene / config.tile_size ** 2)
    n_windows = 10
    ds = SemanticSegmentationWithConfidenceRandomWindowGeoDataset(
        scene,
        out_size=(config.tile_size, config.tile_size),
        # Setting size_lims=(size,size+1) seems weird, but it actually leads to all windows having the same size
        # see https://github.com/azavea/raster-vision/blob/1d23e466d5bbec28373eef5c58efebcb0c774cd1/rastervision_pytorch_learner/rastervision/pytorch_learner/dataset/dataset.py#L408
        size_lims=(config.tile_size, config.tile_size+1),
        padding=None,
        max_windows=n_windows,
        transform=DEFAULT_AUGMENTATIONS,
        normalize=True, # If unsigned integer, bring to range [0, 1]
    )
    #override the sample_window method and within_aoi method
    ds.aoi_centroids = aoi_centroids
    ds.sample_window = custom_sample_window.__get__(ds, SemanticSegmentationWithConfidenceRandomWindowGeoDataset)
    return ds

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

    # print ("\n HERE! Class IDs: ", class_ids)

    # if len(shapes) > 0:
    #     raster = rasterize(
    #         shapes=list(zip(shapes, class_ids)),
    #         out_shape=window.size,
    #         fill=background_class_id,
    #         # merge_alg=merge_max,#MergeAlg.max,  #for overlapping geometries, use the maximum value
    #         dtype=np.float32,
    #         all_touched=all_touched)
    # else:
    #     raster = np.full(window.size, background_class_id, dtype=np.float32)

    #alternative way to rasterize, but it is slower
    # Rasterize each polygon separately
    

    if len(shapes) > 0:
        rasters = []
        #iterate over each geometry in df_int, and rasterize it
        for i in range(len(df_int)):
            # print(i,shapes.iloc[i], class_ids.iloc[i])
            raster = rasterize(
                        shapes = [(shapes.iloc[i], class_ids.iloc[i])],
                        out_shape = window.size,
                        fill = background_class_id,
                        all_touched=all_touched)
            # print(raster)
            rasters.append(raster)  
        #Now merge the rasters such that the maximum value is taken for each pixel
        raster = np.maximum.reduce(rasters)

    else:
        raster = np.full(window.size, background_class_id, dtype=np.float32)
    # print('reduction done: ', raster)

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
    def __init__(self,
                 vector_source: 'VectorSource',
                 background_class_id: int,
                 bbox: Optional['Box'] = None,
                 all_touched: bool = False,
                 raster_transformers: List['RasterTransformer'] = []):
        super().__init__(vector_source, background_class_id, bbox, all_touched, raster_transformers)

    def _get_chip(self, window, out_shape: Optional[Tuple[int, int]] = None):
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
        return np.expand_dims(chip, 2)

class RasterizedConfidences(RasterizedSourceProbability):
    def __init__(self,
                 vector_source: 'VectorSource',
                 background_class_id: int,
                 bbox: Optional['Box'] = None,
                 all_touched: bool = False,
                 raster_transformers: List['RasterTransformer'] = [],
                 other_class_confidence: float = 1.):
        super().__init__(vector_source, background_class_id, bbox, all_touched, raster_transformers)
        self.other_class_confidence = other_class_confidence

    def _get_chip(self, window, out_shape: Optional[Tuple[int, int]] = None):
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
        chip[chip == 0] = self.other_class_confidence
        return np.expand_dims(chip, 2)

class SceneWithConfidences(Scene):
    def __init__(self,
                 id: str,
                 raster_source: 'RasterSource',
                 label_source: Optional['LabelSource'] = None,
                 confidence_source: Optional['LabelSource'] = None,
                 label_store: Optional['LabelStore'] = None,
                 aoi_polygons: Optional[List['BaseGeometry']] = None):
        super().__init__(id, raster_source, label_source, label_store, aoi_polygons)
        self.confidence_source = confidence_source

    def __getitem__(self, key: Any) -> Tuple[Any, Any]:
        x, y = super().__getitem__(key)
        if self.confidence_source is not None:
            confidence = self.confidence_source[key]
        else: 
            confidence = None
        return x, y, confidence
    
def semanticSegmentationWithConfidenceTransformer(
        inp: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
        transform: Optional[A.BasicTransform]):
    x, y, confidence = inp
    assert not (y is None and confidence is None), "Confidence cannot be None when y is None"
    x = np.array(x)
    if transform is not None:
        if y is None and confidence is None:
            x = transform(image=x)['image']
        elif y is not None and confidence is None:
            y = np.array(y)
            out = transform(image=x, mask=y)
            x, y = out['image'], out['mask']
            y = y.astype(int)
        else:
            y, confidence = np.array(y), np.array(confidence)
            out = transform(image=x, mask=y, confidence=confidence)
            x, y, confidence = out["image"], out["mask"], out["confidence"]
            y = y.astype(int)
    return x, y, confidence

def getDataSetItem(dataset, key):
    val = dataset.orig_dataset[key]
    try:
        x, y, confidence = dataset.transform(val)
    except Exception as exc:
        raise exc

    if dataset.normalize and np.issubdtype(x.dtype, np.unsignedinteger):
        max_val = np.iinfo(x.dtype).max
        x = x.astype(float) / max_val

    if dataset.to_pytorch:
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        if y is not None:
            y = torch.from_numpy(y)
        if confidence is not None:
            confidence = torch.from_numpy(confidence)

    if y is None:
        y = torch.tensor(np.nan)
    if confidence is None:
        confidence = torch.tensor(np.nan)

    return x, y, confidence

class SemanticSegmentationWithConfidenceRandomWindowGeoDataset(RandomWindowGeoDataset):
    def __init__(self, 
                 scene: Scene,
                 out_size: Optional[Union[PosInt, Tuple[PosInt, PosInt]]],
                 size_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 h_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 w_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 padding: Optional[Union[NonNegInt, Tuple[NonNegInt,
                                                          NonNegInt]]] = None,
                 max_windows: Optional[NonNegInt] = None,
                 max_sample_attempts: PosInt = 100,
                 return_window: bool = False,
                 efficient_aoi_sampling: bool = True,
                 transform: Optional[A.BasicTransform] = None,
                 normalize: bool = True,
                 to_pytorch: bool = True):
        super().__init__(scene, out_size, size_lims, h_lims, w_lims, padding, 
                         max_windows, max_sample_attempts, return_window, 
                         efficient_aoi_sampling, transform, TransformType.noop, 
                         normalize, to_pytorch)
        self.transform = lambda inp: semanticSegmentationWithConfidenceTransformer(inp, transform)
        
    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration()
        window = self.sample_window()
        if self.return_window:
            return (getDataSetItem(self, window), window)
        return getDataSetItem(self, window)
    
class SemanticSegmentationWithConfidenceSlidingWindowGeoDataset(SlidingWindowGeoDataset):
    def __init__(self, 
                 scene: Scene,
                 out_size: Optional[Union[PosInt, Tuple[PosInt, PosInt]]],
                 size_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 h_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 w_lims: Optional[Tuple[PosInt, PosInt]] = None,
                 padding: Optional[Union[NonNegInt, Tuple[NonNegInt,
                                                          NonNegInt]]] = None,
                 max_windows: Optional[NonNegInt] = None,
                 max_sample_attempts: PosInt = 100,
                 return_window: bool = False,
                 efficient_aoi_sampling: bool = True,
                 transform: Optional[A.BasicTransform] = None,
                 normalize: bool = True,
                 to_pytorch: bool = True):
        super().__init__(scene, out_size, size_lims, h_lims, w_lims, padding, 
                         max_windows, max_sample_attempts, return_window, 
                         efficient_aoi_sampling, transform, TransformType.noop, 
                         normalize, to_pytorch)
        self.transform = lambda inp: semanticSegmentationWithConfidenceTransformer(inp, transform)
        
    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration()
        window = self.sample_window()
        if self.return_window:
            return (getDataSetItem(self, window), window)
        return getDataSetItem(self, window)
