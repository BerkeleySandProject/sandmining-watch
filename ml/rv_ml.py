from rastervision.core.data import SemanticSegmentationLabels, SemanticSegmentationSmoothLabels
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset, SemanticSegmentationLearner,
    SemanticSegmentationGeoDataConfig, SolverConfig, SemanticSegmentationLearnerConfig
)
from .custom_learner import CustomSemanticSegmentationLearner
import torch.nn as nn
from torch.utils.data import Dataset

from experiment_configs.schemas import SupervisedTrainingConfig
from project_config import CLASS_NAME, CLASS_CONFIG


def construct_semantic_segmentation_learner(
        config: SupervisedTrainingConfig,
        model: nn.Module,
        training_ds: Dataset,
        validation_ds: Dataset,
) -> CustomSemanticSegmentationLearner:
    # If a last-model.pth exists in experiment_dir, the learner will load its weights. 
    data_cfg = SemanticSegmentationGeoDataConfig(
        class_names=CLASS_CONFIG.names,
        class_colors=CLASS_CONFIG.colors,
        num_workers=0,
    )
    solver_cfg = SolverConfig(
        batch_sz=config.batch_size,
        lr=config.learning_rate,
        class_loss_weights=[1., config.mine_class_loss_weight]
    )
    learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg)
    learner = CustomSemanticSegmentationLearner(
        experiment_config=config,
        cfg=learner_cfg,
        output_dir=config.output_dir,
        model=model,
        train_ds=training_ds,
        valid_ds=validation_ds,
    )
    return learner       

def predict_site(
        learner: SemanticSegmentationLearner,
        ds: SemanticSegmentationSlidingWindowGeoDataset,
        crop_sz = None
    ) -> SemanticSegmentationSmoothLabels:
    predictions = learner.predict_dataset(
        ds,
        numpy_out=True,
        progress_bar=False,
    )
    predictions = SemanticSegmentationLabels.from_predictions(
        ds.windows,
        predictions,
        smooth=True,
        extent=ds.scene.extent,
        num_classes=len(CLASS_CONFIG),
        crop_sz=crop_sz,
    )
    return predictions

def predict_mine_probability_for_site(
        learner: SemanticSegmentationLearner,
        ds: SemanticSegmentationSlidingWindowGeoDataset,
        crop_sz = None
    ):
    predictions = predict_site(learner, ds, crop_sz)
    scores = predictions.get_score_arr(predictions.extent)
    predicted_mine_probability = scores[CLASS_CONFIG.get_class_id(CLASS_NAME)]
    return predicted_mine_probability

def predict_class_for_site(
        learner: SemanticSegmentationLearner,
        ds: SemanticSegmentationSlidingWindowGeoDataset,
        crop_sz = None
    ):
    predictions = predict_site(learner, ds, crop_sz)
    predicted_class = predictions.get_label_arr(predictions.extent, null_class_id=-1)
    return predicted_class
