import numpy as np
from rastervision.core.data import (
    ClassInferenceTransformer, GeoJSONVectorSource,
    RasterioSource, RasterizedSource, Scene, SemanticSegmentationLabelSource,
    CastTransformer, SemanticSegmentationLabels
)
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset, SemanticSegmentationLearner,
    SemanticSegmentationGeoDataConfig, SolverConfig, SemanticSegmentationLearnerConfig
)
import torch.nn as nn
from torch.utils.data import Dataset

from project_config import CLASS_NAME, CLASS_CONFIG
from utils.schemas import ObservationPointer


def observation_to_scene(observation: ObservationPointer, channels) -> Scene:
    return create_scene(
        observation.uri_to_bs,
        observation.uri_to_annotations,
        observation.name,
        channels,
    )

def create_scene(img_uri, label_uri, scene_id, channels) -> Scene:
    img_raster_source = RasterioSource(
        img_uri,
        channel_order=channels,
        allow_streaming=False,
        raster_transformers=[CastTransformer(np.uint16)]
    )

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

def scene_to_validation_ds(scene: Scene, tile_size: int):
    # No augementation and windows don't overlap
    return SemanticSegmentationSlidingWindowGeoDataset(
        scene,
        size=tile_size,
        stride=tile_size,
        padding=tile_size,
        pad_direction='end',
        transform=None,
    )

def scene_to_training_ds(scene: Scene, tile_size: int, augmentation):
    # Has augementation and overlapping windows
    return SemanticSegmentationSlidingWindowGeoDataset(
        scene,
        size=tile_size,
        stride=int(tile_size / 2),
        padding=None,
        pad_direction='both',
        transform=augmentation,
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
    ):
    predictions = learner.predict_dataset(
        ds,
        raw_out=True,
        numpy_out=True,
        progress_bar=True
    )
    pred_labels = SemanticSegmentationLabels.from_predictions(
        ds.windows,
        predictions,
        smooth=True,
        extent=ds.scene.extent,
        num_classes=len(CLASS_CONFIG)
    )
    scores = pred_labels.get_score_arr(pred_labels.extent)
    predicted_mine_probability = scores[CLASS_CONFIG.get_class_id(CLASS_NAME)]
    return predicted_mine_probability
