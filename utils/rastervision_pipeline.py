import numpy as np
from rastervision.core.data import (
    ClassInferenceTransformer, GeoJSONVectorSource,
    RasterioSource, MultiRasterSource, RasterizedSource, Scene,
    SemanticSegmentationLabelSource
)
from rastervision.core.data.raster_transformer.nan_transformer import NanTransformer

from rastervision.core.data.raster_source import RasterSource
from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset


from project_config import CLASS_NAME, CLASS_CONFIG
from experiment_configs.schemas import SupervisedTrainingConfig, DatasetChoice
from utils.schemas import ObservationPointer
from ml.norm_data import NormS1Transformer, s2_norm_transformer_config_map


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
    transformers_norming = s2_norm_transformer_config_map[config.s2_norming]
    return RasterioSource(
        img_uri,
        channel_order=config.s2_channels,
        allow_streaming=False,
        raster_transformers=transformers_norming
    )

def create_s1_image_source(config, img_uri):
    return RasterioSource(
        img_uri,
        channel_order=None,
        allow_streaming=False,
        raster_transformers=[
            NanTransformer(),
            NormS1Transformer()
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
        padding=0,
        pad_direction='end',
        transform=None,
    )

def scene_to_training_ds(config, scene: Scene):
    # Has augementation and overlapping windows
    return SemanticSegmentationSlidingWindowGeoDataset(
        scene,
        size=config.tile_size,
        stride=int(config.tile_size / 2),
        padding=0,
        pad_direction='end',
        transform=config.augmentations,
    )

def scene_to_prediction_ds(config, scene: Scene):
    # No augmentation and overlapping windows
    return SemanticSegmentationSlidingWindowGeoDataset(
        scene,
        size=config.tile_size,
        stride=int(config.tile_size / 2),
        padding=None,
        pad_direction='both',
        transform=None,
    )
