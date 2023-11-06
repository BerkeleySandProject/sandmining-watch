import numpy as np
from sklearn.metrics import average_precision_score
import wandb

from utils.wandb_utils import create_semantic_segmentation_image

from project_config import N_EDGE_PIXELS_DISCARD


def evaluate_predicitions(prediction_results):
    """
    Expects prediction_results to be a list of dictionaries. Each dictonary is expected
    to have the keys 'predictions', 'ground_truth' and 'name'
    """
    all_predictions_list = []
    all_gt_list = []

    eval_dict = {}
    for prediction_result_dict in prediction_results:
        # We discard the edge pixels in each observation. This ensures that 
        # results are comparable, even if the window size is different.
        # (But N_EDGE_PIXELS_DISCARD needs to remain consistent)
        prediction = center_crop(prediction_result_dict['predictions']).ravel()
        gt = center_crop(prediction_result_dict['ground_truth']).ravel()
        observation_name = prediction_result_dict['name']

        all_predictions_list.append(prediction)
        all_gt_list.append(gt)

        ground_truth_is_all_negative = np.all(gt == 0)
        if ground_truth_is_all_negative:
            # If no ground truth element is positive, it makes no sense to compute our metrics,
            # because we will have no true positives.
            continue

        precision, recall, f1_score, average_precision = compute_metrics(gt, prediction)
        eval_dict.update({
            f"eval/{observation_name}/precision": precision,
            f"eval/{observation_name}/recall": recall,
            f"eval/{observation_name}/f1_score": f1_score,
            f"eval/{observation_name}/average_precision": average_precision,
        })

    all_predictions = np.concatenate(all_predictions_list)
    all_gt = np.concatenate(all_gt_list)
    precision, recall, f1_score, average_precision = compute_metrics(all_gt, all_predictions)
    eval_dict.update({
        f"eval/total/precision": precision,
        f"eval/total/recall": recall,
        f"eval/total/f1_score": f1_score,
        f"eval/total/average_precision": average_precision,
    })
    return eval_dict

def make_wandb_segmentation_masks(prediction_results):
    """
    Returns list of wandb.Image. Background is RGB image + ground truth and prediction mask.

    Expects prediction_results to be a list of dictionaries. Each dictonary is expected
    to have the keys 'predictions', 'ground_truth', 'rgb_img' and 'name'.
    """
    def prediction_results_to_wandb_mask(input):
        return create_semantic_segmentation_image(
            background_image=center_crop(input['rgb_img']),
            predicted_mask=center_crop(input['predictions']) > 0.5,
            ground_truth_mask=input['ground_truth'],
            image_title=input['name']
        )    
    return list(map(prediction_results_to_wandb_mask, prediction_results))

def make_wandb_predicted_probs_images(prediction_results):
    """
    Returns list of wandb.Image with prediction probabilities

    Expects prediction_results to be a list of dictionaries. Each dictonary is expected
    to have the keys 'predictions', 'ground_truth', 'rgb_img' and 'name'.
    """
    def create_predicted_probabilities_image(input):
        return wandb.Image(
            center_crop(input['predictions']),
            caption=input['name']
        )
    return list(map(create_predicted_probabilities_image, prediction_results))

def center_crop(array, n_crop_pixels=N_EDGE_PIXELS_DISCARD):
    """
    Given a 2-dimensional array, crops this array by n_crop_pixels at each edge
    """
    return array[n_crop_pixels:-n_crop_pixels, n_crop_pixels:-n_crop_pixels]

def compute_metrics(ground_truth, predicted_prob):
    predicted_class = predicted_prob > 0.5

    TP = ((predicted_class == 1) & (ground_truth == 1)).sum()  # True Positives
    FP = ((predicted_class == 1) & (ground_truth == 0)).sum()  # False Positives
    FN = ((predicted_class == 0) & (ground_truth == 1)).sum()  # False Negatives
    TN = ((predicted_class == 0) & (ground_truth == 0)).sum()  # True Negatives

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    average_precision = average_precision_score(ground_truth, predicted_prob)

    return precision, recall, f1_score, average_precision
