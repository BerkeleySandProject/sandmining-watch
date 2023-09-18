import wandb
from project_config import CLASS_CONFIG

def create_semantic_segmentation_image(
        background_image,
        predicted_mask,
        ground_truth_mask,
        image_title,
):    
    labels_dict = {
        CLASS_CONFIG.get_class_id(name): name for name in CLASS_CONFIG.names
    }
    wand_image_with_masks = wandb.Image(
        background_image,
        masks={
            "Prediction": {
                "mask_data": predicted_mask,
                "class_labels": labels_dict
            },
            "Ground truth": {
                "mask_data": ground_truth_mask,
                "class_labels": labels_dict
            },
        },
        caption=image_title,
    )
    return wand_image_with_masks
