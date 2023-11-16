from typing import Any, List, Tuple, Optional, Union
import albumentations as A
import numpy as np
from math import ceil
from rastervision.core.box import Box
from rastervision.core.data import (
    ClassInferenceTransformer, GeoJSONVectorSource, RasterSource, RasterTransformer,
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


from project_config import CLASS_NAME, CLASS_CONFIG, USE_RIVER_AOIS, HARD_LABELS
from experiment_configs.schemas import (
    SupervisedTrainingConfig, DatasetChoice, NormalizationS2Choice, 
    DatasetsConfig, LabelChoice, ConfidenceChoice
)
from project_config import CLASS_NAME, CLASS_CONFIG, USE_RIVER_AOIS, N_EDGE_PIXELS_DISCARD
from experiment_configs.schemas import SupervisedTrainingConfig, DatasetChoice, NormalizationS2Choice
from utils.schemas import ObservationPointer
from ml.norm_data import norm_s1_transformer, norm_s2_transformer, divide_by_10000_transformer
from ml.augmentations import DEFAULT_AUGMENTATIONS

from rasterio.features import rasterize, MergeAlg
import geopandas as gpd

from typing import TYPE_CHECKING, Optional, List
from shapely.geometry import Polygon

def observation_to_scene(config: SupervisedTrainingConfig, observation: ObservationPointer) -> Scene:
    if config.datasets.images == DatasetChoice.S1S2:
        return create_scene_s1s2(
            config,
            s2_uri=observation.uri_to_s2,
            s1_uri=observation.uri_to_s1,
            label_uri=observation.uri_to_annotations,
            scene_id=observation.name,
            rivers_uri=observation.uri_to_rivers
        )
    elif config.datasets.images == DatasetChoice.S2:
        return create_scene_s2(
            config,
            s2_uri=observation.uri_to_s2,
            label_uri=observation.uri_to_annotations,
            scene_id=observation.name,
            rivers_uri=observation.uri_to_rivers
        )
    elif config.datasets.images == DatasetChoice.S2_L1C:
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
        raw_image = raster_source.get_raw_chip(raster_source.extent)
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
        label_source = SemanticSegmentationLabelSource(
            RasterizedSource(
                vector_source,
                background_class_id=CLASS_CONFIG.null_class_id,
                bbox=img_raster_source.bbox), 
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
        scene = ThreeClassScene(id=scene_id, 
                        raster_source=img_raster_source, 
                        label_source=label_source)

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

def scene_to_validation_ds(config, scene: Scene):
    # No augementation and windows don't overlap. Use for validation during training time.
    ds = SemanticSegmentationSlidingWindowGeoDatasetCustom(
        scene=scene,
        ignore_aoi=False, # Filter windows by AOI 
        size=config.tile_size,
        stride=config.tile_size,
        padding=config.tile_size,
        pad_direction='end',
        transform=None,
        normalize=True, # If unsigned integer, bring to range [0, 1]
    )
    return ds


def scene_to_inference_ds(config, scene: Scene, full_image: bool):
    # In inference mode, we have a slining window configuration with overlapping windows.
    # If full_image is True, sliding windows span the entire scene. Otherwise, sliding windows
    # are filtered by the scene's AOI (if existent)
    stride = config.tile_size - N_EDGE_PIXELS_DISCARD * 2
    ds = SemanticSegmentationSlidingWindowGeoDatasetCustom(
        scene=scene,
        ignore_aoi=full_image, # If we want to run inference on the full image, we ignore the AOI.
        size=config.tile_size,
        stride=stride,
        padding=config.tile_size,
        pad_direction='end',
        transform=None,
        normalize=True, # If unsigned integer, bring to range [0, 1]
    )
    return ds

    
from rastervision.core.box import Box
from typing import List
from shapely.geometry import Polygon

def centroid_within_polygons(window: 'Box', aoi_polygons: List[Polygon]) -> bool:
    """Check if window's centroid is within a list of AOI polygons."""

    w = window.to_shapely()
    for polygon in aoi_polygons:
        if w.centroid.within(polygon):
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
            if centroid_within_polygons(window, self.aoi_polygons):
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
    ds = ThreeClassSemanticSegmentationRandomWindowGeoDataset(
        scene,
        out_size=(config.tile_size, config.tile_size),
        # Setting size_lims=(size,size+1) seems weird, but it actually leads to all windows having the same size
        # see https://github.com/azavea/raster-vision/blob/1d23e466d5bbec28373eef5c58efebcb0c774cd1/rastervision_pytorch_learner/rastervision/pytorch_learner/dataset/dataset.py#L408
        size_lims=(config.tile_size, config.tile_size+1),
        padding=None,
        max_windows=n_windows,
        transform=DEFAULT_AUGMENTATIONS,
        normalize=True, # If unsigned integer, bring to range [0, 1],
        # When using our custom sampling stragety that checks for window's centroids,
        # efficient_aoi_sampling needs to be turned off.
        efficient_aoi_sampling=(not aoi_centroids),
    )
    #override the sample_window method and within_aoi method
    ds.aoi_centroids = aoi_centroids
    ds.sample_window = custom_sample_window.__get__(ds, ThreeClassSemanticSegmentationRandomWindowGeoDataset)
    return ds

class ThreeClassScene(Scene):
    def __init__(self,
                 id: str,
                 raster_source: 'RasterSource',
                 label_source: Optional['LabelSource'] = None,
                 label_store = None,
                 aoi_polygons: Optional[List['BaseGeometry']] = None):
        super().__init__(id, raster_source, label_source, label_store, aoi_polygons)
    
    def __getitem__(self, key: Any) -> Tuple[Any, Any]:
        x, y = super().__getitem__(key)
        w = y.copy()
        w[y == 2] = 0
        w[y != 2] = 1
        return x, y, w
    
def threeClassSemanticSegmentationTransformer(
        inp: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
        transform: Optional[A.BasicTransform]):
    x, y, w = inp
    x = np.array(x)
    if transform is not None:
        if y is None and w is None:
            x = transform(image=x)['image']
        elif y is not None and w is None:
            y = np.array(y)
            out = transform(image=x, mask=y)
            x, y = out['image'], out['mask']
            y = y.astype(int)
        else:
            y, w = np.array(y), np.array(w)
            out = transform(image=x, mask=y, weights=w)
            x, y, w = out["image"], out["mask"], out["weights"]
            y = y.astype(int)
    return x, y, w


def getDataSetItem(dataset, key):
    val = dataset.orig_dataset[key]
    try:
        x, y, w = dataset.transform(val)
    except Exception as exc:
        raise exc

    if dataset.normalize and np.issubdtype(x.dtype, np.unsignedinteger):
        max_val = np.iinfo(x.dtype).max
        x = x.astype(float) / max_val

    if dataset.to_pytorch:
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        if y is not None:
            y = torch.from_numpy(y)
        if w is not None:
            w = torch.from_numpy(w)

    if y is None:
        y = torch.tensor(np.nan)
    if w is None:
        w = torch.tensor(np.nan)

    return x, y, w

class ThreeClassSemanticSegmentationRandomWindowGeoDataset(RandomWindowGeoDataset):
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
        self.transform = lambda inp: threeClassSemanticSegmentationTransformer(inp, transform)
        
    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration()
        window = self.sample_window()
        if self.return_window:
            return (getDataSetItem(self, window), window)
        return getDataSetItem(self, window)
    
class ThreeClassSemanticSegmentationSlidingWindowGeoDataset(SlidingWindowGeoDataset):
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
        self.transform = lambda inp: threeClassSemanticSegmentationTransformer(inp, transform)
        
    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration()
        window = self.sample_window()
        if self.return_window:
            return (getDataSetItem(self, window), window)
        return getDataSetItem(self, window)

class SemanticSegmentationSlidingWindowGeoDatasetCustom(SemanticSegmentationSlidingWindowGeoDataset):
    """
    The default SemanticSegmentationSlidingWindowGeoDataset requires windows to lie complement within in
    the AOI. This laternative relaxes the conditation. Windows are accepted which has an overlap with the AOI

    An additional parameter allows to ignore to AOI and sample windows across the entire image.
    """
    def __init__(self, ignore_aoi = False, **kwargs):
        self.ignore_aoi = ignore_aoi
        super().__init__(**kwargs)
    def init_windows(self) -> None:
        """Pre-compute windows."""
        windows = self.scene.raster_source.extent.get_windows(
            self.size,
            stride=self.stride,
            padding=self.padding,
            pad_direction=self.pad_direction)
        if len(self.scene.aoi_polygons_bbox_coords) > 0 and not self.ignore_aoi:
            windows = Box.filter_by_aoi(windows,
                                        self.scene.aoi_polygons_bbox_coords,
                                        within=False)
        self.windows = windows
