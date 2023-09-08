from rastervision.core.data import SemanticSegmentationLabels
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset, SemanticSegmentationLearner,
    SemanticSegmentationGeoDataConfig, SolverConfig, SemanticSegmentationLearnerConfig
)
from .custom_learner import CustomSemanticSegmentationLearner
import torch.nn as nn
from torch.utils.data import Dataset

from project_config import CLASS_NAME, CLASS_CONFIG


def construct_semantic_segmentation_learner(
        model: nn.Module,
        training_ds: Dataset,
        validation_ds: Dataset,
        batch_size,
        learning_rate,
        experiment_dir,
        class_loss_weights=None,
) -> CustomSemanticSegmentationLearner:
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
    learner = CustomSemanticSegmentationLearner(
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
