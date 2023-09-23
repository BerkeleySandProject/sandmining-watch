from rastervision.core.data import SemanticSegmentationLabels, SemanticSegmentationSmoothLabels
from rastervision.pytorch_learner import (
    SemanticSegmentationSlidingWindowGeoDataset, SemanticSegmentationLearner,
)

from project_config import CLASS_NAME, CLASS_CONFIG
     

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
