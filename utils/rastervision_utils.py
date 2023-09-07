import numpy as np
from rastervision.core.data import (
    ClassInferenceTransformer, GeoJSONVectorSource,
    RasterioSource, MultiRasterSource, RasterizedSource, Scene,
    SemanticSegmentationLabels, SemanticSegmentationLabelSource
)

from rastervision.core.data.raster_transformer.nan_transformer import NanTransformer

from rastervision.core.data.raster_source import RasterSource
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset, SemanticSegmentationLearner,
    SemanticSegmentationGeoDataConfig, SolverConfig, SemanticSegmentationLearnerConfig
)
import torch.nn as nn
from torch.utils.data import Dataset

from project_config import CLASS_NAME, CLASS_CONFIG
from experiment_config import SupervisedTrainingConfig, DatasetChoice, S2NormChoice
from utils.schemas import ObservationPointer
from utils.sandmining_ml_utils import NormS1Transformer, NormS2Transformer, SatMaeNormS2Transformer


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
    transformer_config_map = {
        S2NormChoice.SatMAE: [
            NanTransformer(),
            SatMaeNormS2Transformer(),
        ],
        S2NormChoice.Div: [
            NanTransformer(),
            NormS2Transformer(),
        ]
    }
    return RasterioSource(
        img_uri,
        channel_order=config.s2_channels,
        allow_streaming=False,
        raster_transformers=transformer_config_map[config.s2_norming]
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
        padding=config.tile_size,
        pad_direction='end',
        transform=None,
    )

def scene_to_training_ds(config, scene: Scene):
    # Has augementation and overlapping windows
    return SemanticSegmentationSlidingWindowGeoDataset(
        scene,
        size=config.tile_size,
        stride=int(config.tile_size / 2),
        padding=None,
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

def construct_semantic_segmentation_learner(
        model: nn.Module,
        training_ds: Dataset,
        validation_ds: Dataset,
        batch_size,
        learning_rate,
        experiment_dir,
        class_loss_weights=None,
) -> SemanticSegmentationLearner:
    # If a last-model.pth exists in experiment_dir, the learner will load its weights. 
    data_cfg = SemanticSegmentationGeoDataConfig(
        class_names=CLASS_CONFIG.names,
        class_colors=CLASS_CONFIG.colors,
        num_workers=0,
    )
    solver_cfg = SolverConfig(
        batch_sz=batch_size,
        lr=learning_rate,
        class_loss_weights=class_loss_weights
    )
    learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg)
    learner = SemanticSegmentationLearner(
        cfg=learner_cfg,
        output_dir=experiment_dir,
        model=model,
        train_ds=training_ds,
        valid_ds=validation_ds,
    )
    return learner       


def get_predictions_for_site(
        learner: SemanticSegmentationLearner,
        ds: SemanticSegmentationSlidingWindowGeoDataset,
        crop_sz = None
    ):
    predictions = learner.predict_dataset(
        ds,
        raw_out=True,
        numpy_out=True,
        progress_bar=False,
    )
    pred_labels = SemanticSegmentationLabels.from_predictions(
        ds.windows,
        predictions,
        smooth=True,
        extent=ds.scene.extent,
        num_classes=len(CLASS_CONFIG),
        crop_sz=crop_sz
    )
    scores = pred_labels.get_score_arr(pred_labels.extent)
    predicted_mine_probability = scores[CLASS_CONFIG.get_class_id(CLASS_NAME)]
    return predicted_mine_probability
