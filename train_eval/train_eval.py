"""
CLI Tool for training and evaluating models.
"""

from argparse import ArgumentParser
import os
import sys
from dotenv import load_dotenv
from google.cloud import storage
from torch.utils.data import ConcatDataset
import json
import ipdb
import wandb
import numpy as np
import random

sys.path.insert(0, os.path.abspath(".."))

__author__ = "Sushant Harischandra Vema"
__version__ = "0.1.0"


load_dotenv()


def main(_):
    from project_config import GCP_PROJECT_NAME
    from project_config import DATASET_JSON_PATH
    from project_config import NUM_EPOCHS
    from project_config import RUN_NAME

    gcp_client = storage.Client(project=GCP_PROJECT_NAME)
    from utils.rastervision_pipeline import GoogleCloudFileSystem

    GoogleCloudFileSystem.storage_client = gcp_client
    from experiment_configs.configs import (
        satlas_swin_base_si_ms_linear_decoder_config,
        lora_config,
    )
    from ml.learner_factory import learner_factory
    import torch

    # For reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Configuration
    config = satlas_swin_base_si_ms_linear_decoder_config

    # Create Rastervision Datasets
    from utils.rastervision_pipeline import (
        observation_to_scene,
        scene_to_training_ds,
        scene_to_validation_ds,
        scene_to_inference_ds,
        warn_if_nan_in_raw_raster,
    )
    from utils.data_management import observation_factory, characterize_dataset

    root_dir = os.getcwd()  # get the current working directory
    json_rel_path = (
        "../" + DATASET_JSON_PATH
    )  # define the relative path to the dataset JSON
    json_abs_path = os.path.join(
        root_dir, json_rel_path
    )  # Combine root directory with the relative path

    dataset_json = json.load(open(json_abs_path, "r"))

    VALIDATION_CLUSTER_ID = max(
        [observation["cluster_id"] for observation in dataset_json]
    )
    all_observations = observation_factory(dataset_json)

    training_scenes = []
    validation_scenes = []

    for observation in all_observations:
        if observation.cluster_id == VALIDATION_CLUSTER_ID:
            validation_scenes.append(observation_to_scene(config, observation))
        else:
            training_scenes.append(observation_to_scene(config, observation))

    print('Using "spatially-random" validation cluster id: ', VALIDATION_CLUSTER_ID)

    training_datasets = [
        # random window sampling happens here
        scene_to_training_ds(config, scene)
        for scene in training_scenes
    ]
    validation_datasets = [
        # scene_to_validation_ds(config, scene) for scene in validation_scenes
        # better performance with this
        scene_to_inference_ds(
            config, scene, full_image=False, stride=int(config.tile_size / 2)
        )
        for scene in validation_scenes
    ]
    # DEBUG: Check training and validation datasets
    train_dataset_merged = ConcatDataset(training_datasets)
    val_dataset_merged = ConcatDataset(validation_datasets)

    print("Training dataset size: {:4d} images".format(
        len(train_dataset_merged)))
    print("Testing dataset size: {:4d}  images".format(
        len(val_dataset_merged)))

    mine_percentage_aoi = characterize_dataset(
        training_scenes, validation_scenes)
    # Update mine class weight
    # config.mine_class_loss_weight = 10
    # print("Updating mine class weight:", config.mine_class_loss_weight)

    # Train
    from models.model_factory import model_factory, print_trainable_parameters
    from ml.optimizer_factory import optimizer_factory
    from ml.model_stats import count_number_of_weights
    from ml.learner import BinarySegmentationLearner, MultiSegmentationLearner
    from utils.rastervision_pipeline import scene_to_inference_ds

    _, _, n_channels = training_datasets[0].scene.raster_source.shape

    model = model_factory(config, n_channels=n_channels,
                          config_lora=lora_config)

    print("Before LoRA: ")
    all_params, trainable_params = count_number_of_weights(model)
    print(
        f"trainable_params: {trainable_params/1e6}M | | all params: {all_params/1e6}M | | trainable %: {100 * trainable_params // all_params: .2f}"
    )

    optimizer = optimizer_factory(config, model)

    learner = learner_factory(
        config=config,
        model=model,
        optimizer=optimizer,
        # for development and debugging, use training_datasets[0] or similar to speed up
        train_ds=train_dataset_merged,
        # for development and debugging, use training_datasets[1] or similar to speed up
        valid_ds=val_dataset_merged,
        output_dir=(
            "~/sandmining-watch/out/OUTPUT_DIR/satlas_spatial_random_lora"),
        save_model_checkpoints=True,
    )

    print_trainable_parameters(learner.model)

    learner.log_data_stats()

    # Run this if you want to log the run to weights and biases
    learner.initialize_wandb_run(run_name=RUN_NAME)
    print(f"Number of epochs: {NUM_EPOCHS}")
    learner.train(epochs=NUM_EPOCHS)

    def evaluate():
        # Evaluate
        # ipdb.set_trace()
        # DEBUG: Check successful training before evaluation
        from ml.learner import BinarySegmentationPredictor, MultiSegmentationLearner
        from utils.rastervision_pipeline import scene_to_inference_ds

        # evaluation_datasets =  [
        #     scene_to_inference_ds(
        #         config, scene, full_image=True
        #     ) for scene in validation_scenes
        # ]
        evaluation_datasets = []
        all_observations = observation_factory(dataset_json)
        for observation in all_observations:
            if (
                observation.cluster_id == VALIDATION_CLUSTER_ID
            ):  # statically assign clusetr zero to validation set
                evaluation_datasets.append(
                    scene_to_inference_ds(
                        config,
                        observation_to_scene(
                            config, observation, weights_class=False),
                        full_image=True,
                    )
                )

        predictor = BinarySegmentationPredictor(
            config,
            model,
        )

        # # Alternatively: specify path to trained weights
        # path_to_weights = expanduser("~/sandmining-watch/out/1102-satmae-1/last-model.pth")
        # predictor = BinarySegmentationPredictor(
        #     config,
        #     model,
        #     path_to_weights,
        # )

        from ml.eval_utils import (
            evaluate_predicitions,
            make_wandb_segmentation_masks,
            make_wandb_predicted_probs_images,
        )
        from utils.visualizing import raster_source_to_rgb

        prediction_results_list = []

        for ds in evaluation_datasets[:1]:
            predictions = predictor.predict_mine_probability_for_site(ds)

            rgb_img = raster_source_to_rgb(ds.scene.raster_source)
            prediction_results_list.append(
                {
                    "predictions": predictions,
                    "ground_truth": ds.scene.label_source.get_label_arr(),
                    "rgb_img": rgb_img,
                    "name": ds.scene.id,
                }
            )

        evaluation_results_dict = evaluate_predicitions(
            prediction_results_list)

        assert wandb.run is not None

        # Add lists of W&B images to dict
        evaluation_results_dict.update(
            {
                "Segmenation masks": make_wandb_segmentation_masks(
                    prediction_results_list
                ),
                "Predicted probabilites": make_wandb_predicted_probs_images(
                    prediction_results_list
                ),
            }
        )

        # Log to W&B
        wandb.log(evaluation_results_dict)

        return

    wandb.finish


# def visualize():
#     from utils.visualizing import visualize_dataset
#     import torch
#
#     # for ds in training_datasets:
#     #       visualize_dataset(ds)
#     torch.manual_seed(42)
#     visualize_dataset([training_datasets[0]])
#
#     # for ds in validation_datasets:
#     #     visualize_dataset(ds)
#     return
#

if __name__ == "__main__":
    # Required positional argument
    parser = ArgumentParser()

    # Save to Weights and Biases?
    parser.add_argument(
        "-w",
        "--weightsbias",
        action="store_true",
        default=False,
        help="Toggle logging results to weights and biases.",
    )

    # Model Name
    # parser.add_argument(
    #     "-m",
    #     "--model_name",
    #     default="test",
    #     const="all",
    #     nargs="?",
    #     choices=[
    #         "test",
    #         "unet",
    #         "segformer",
    #         "ssl4eo_resnet18",
    #         "ssl4eo_resnet50",
    #         "satmae_base",
    #         "satmae_large",
    #         "satlas_base",
    #     ],
    #     help="Pick the name of the model you want to train.",
    # )
    #
    # Select Mode
    parser.add_argument(
        "--mode",
        const="all",
        nargs="?",
        choices=["train", "evaluate", "visualize"],
        help="Choose to either. (default: %(default))",
    )

    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity (-v, -vv, etc)"
    )

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__),
    )
    _ = parser.parse_args()
    main(_)
