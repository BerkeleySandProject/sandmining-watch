import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import wandb

from utils.wandb_utils import create_semantic_segmentation_image

from project_config import N_EDGE_PIXELS_DISCARD


def evaluate_predictions(prediction_results):
    """
    Expects prediction_results to be a list of dictionaries. Each dictonary is expected
    to have the keys 'predictions', 'ground_truth', 'name' and 'crop_sz' (in pixels).
    """
    all_predictions_list = []
    all_gt_list = []
    best_thresholds = []

    eval_dict = {}
    for prediction_result_dict in prediction_results:
        # We discard the edge pixels in each observation. This ensures that 
        # results are comparable, even if the window size is different.
        # (But N_EDGE_PIXELS_DISCARD needs to remain consistent)
        
        prediction = center_crop(prediction_result_dict['predictions'], prediction_result_dict['crop_sz']).ravel()
        gt = center_crop(prediction_result_dict['ground_truth'], prediction_result_dict['crop_sz']).ravel()
        observation_name = prediction_result_dict['name']
        #also crop the rgb image so that the dimensions match with predictions and gt
        prediction_result_dict['rgb_img'] = center_crop(prediction_result_dict['rgb_img'], prediction_result_dict['crop_sz'])

        all_predictions_list.append(prediction)
        all_gt_list.append(gt)

        ground_truth_is_all_negative = np.all(gt == 0)
        if ground_truth_is_all_negative:
            # If no ground truth element is positive, it makes no sense to compute our metrics,
            # because we will have no true positives.
            continue

        precision, recall, f1_score, average_precision, best_threshold, best_f1_score = compute_metrics(gt, prediction)
        best_thresholds.append(best_threshold)

        eval_dict.update({
            f"eval/{observation_name}/precision": precision,
            f"eval/{observation_name}/recall": recall,
            f"eval/{observation_name}/f1_score": f1_score,
            f"eval/{observation_name}/average_precision": average_precision,
            f"eval/{observation_name}/best_threshold": best_threshold,
            f"eval/{observation_name}/best_f1_score": best_f1_score,
        })

    all_predictions = np.concatenate(all_predictions_list)
    all_gt = np.concatenate(all_gt_list)

    mean_threshold = np.mean(best_thresholds)
    median_threshold = np.median(best_thresholds)
    sd_threshold = np.std(best_thresholds)

    precision, recall, f1_score, average_precision, _, _,  = compute_metrics(all_gt, all_predictions)
    eval_dict.update({
        f"eval/total/precision": precision,
        f"eval/total/recall": recall,
        f"eval/total/f1_score": f1_score,
        f"eval/total/average_precision": average_precision,
        f"eval/total/mean_threshold": mean_threshold,
        f"eval/total/median_threshold": median_threshold,
        f"eval/total/sd_threshold": sd_threshold,
    })

    #replace nans in all_gt and all_predictions with 0
    all_gt = np.nan_to_num(all_gt)
    all_predictions = np.nan_to_num(all_predictions)


    if wandb.run is not None:
        pr_curve = make_precision_recall_curve_plot(all_gt, all_predictions)
        eval_dict.update({
            f"eval/total/precision_recall_curve": pr_curve,
        })
    return eval_dict


def make_precision_recall_curve_plot(gt, predictions):
    # wandb.plot.pr_curve() expects the predictions to have the shape (*y_true.shape, num_classes)
    # -> hack: we duplicate the array of shape (n,) into (n,2)
    preds = np.stack([predictions, predictions], axis=1)
    return wandb.plot.pr_curve(
        gt, preds, classes_to_plot=[1], labels=['other', 'sandmine']
    )

def make_wandb_segmentation_masks(prediction_results, threshold=0.5):
    """
    Returns list of wandb.Image. Background is RGB image + ground truth and prediction mask.

    Expects prediction_results to be a list of dictionaries. Each dictonary is expected
    to have the keys 'predictions', 'ground_truth', 'rgb_img' and 'name'.
    """
    def prediction_results_to_wandb_mask(input):
        return create_semantic_segmentation_image(
            background_image=center_crop(input['rgb_img']),
            predicted_mask=center_crop(input['predictions']) >= threshold,
            ground_truth_mask=center_crop(input['ground_truth']),
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

def center_crop(array, n_crop_pixels=0):
# def center_crop(array, n_crop_pixels=N_EDGE_PIXELS_DISCARD):
    """
    Given a 2-dimensional array, crops this array by n_crop_pixels at each edge
    """
    return array[n_crop_pixels:-n_crop_pixels, n_crop_pixels:-n_crop_pixels] if n_crop_pixels > 0 else array

def compute_metrics(ground_truth, predicted_prob, predicted_class=None):
    eps = 1e-6
    valid_mask = ~np.isnan(predicted_prob)  # Create a mask for valid (non-nan) values

    # print("Number of valid pixels: ",valid_mask.sum())

    # Apply the mask to ground_truth and predicted_prob
    valid_ground_truth = ground_truth[valid_mask]
    valid_predicted_prob = predicted_prob[valid_mask]

    if predicted_class is None:
        predicted_class = valid_predicted_prob > 0.5
    else:
        predicted_class = predicted_class[valid_mask]

    TP = ((predicted_class == 1) & (valid_ground_truth == 1)).sum()  # True Positives
    FP = ((predicted_class == 1) & (valid_ground_truth == 0)).sum()  # False Positives
    FN = ((predicted_class == 0) & (valid_ground_truth == 1)).sum()  # False Negatives
    TN = ((predicted_class == 0) & (valid_ground_truth == 0)).sum()  # True Negatives

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1_score = 2 * (precision * recall) / (precision + recall)

    precisions_t, recalls_t, thresholds = precision_recall_curve(valid_ground_truth, valid_predicted_prob)
    f1_scores_t = 2 * (precisions_t * recalls_t) / (precisions_t + recalls_t + eps)

    average_precision = average_precision_score(valid_ground_truth, valid_predicted_prob)

    #get best threshold
    ix = np.argmax(f1_scores_t)
    return precision, recall, f1_score, average_precision, thresholds[ix], f1_scores_t[ix]


import rasterio, os
from rasterio.transform import from_origin

def save_mask_as_geotiff(mask, transform, crs, filename):
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=str(mask.dtype),
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

def save_predictions(
        prediction,
        path,
        class_config,
        crs_transformer,
        threshold=0.5
    ):
    
    prediction.save(path, class_config=class_config, crs_transformer=crs_transformer, discrete_output=False)

    prediction_score = prediction.get_score_arr(prediction.extent)[0]

    mask = np.where(prediction_score >= threshold, 1, 0)

    mask_filename = os.path.join(path, 'mask.tif')
    save_mask_as_geotiff(mask, crs_transformer.transform, crs='EPSG:4326', filename=mask_filename)