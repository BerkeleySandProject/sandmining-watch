import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from os.path import expanduser

from dotenv import load_dotenv
load_dotenv()


from models.model_factory import model_factory, print_trainable_parameters
from ml.optimizer_factory import optimizer_factory
from ml.learner_factory import learner_factory
import os, torch
import numpy as np
from torch.utils.data import ConcatDataset
import json
from utils.rastervision_pipeline import observation_to_scene, scene_to_training_ds, scene_to_inference_ds
import random
from google.cloud import storage
from utils.rastervision_pipeline import GoogleCloudFileSystem
from project_config import GCP_PROJECT_NAME, DATASET_JSON_PATH
from experiment_configs.configs import lora_config, satmae_large_config_lora_methodA, satmae_large_config_lora_methodB
from torch.utils.data import ConcatDataset
from utils.data_management import observation_factory
from datetime import datetime


gcp_client = storage.Client(project=GCP_PROJECT_NAME)


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" #to prevent cuda out of memory error
torch.cuda.empty_cache()

#For reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

methodA = True

if methodA:
    config = satmae_large_config_lora_methodA
else:
    config = satmae_large_config_lora_methodB

# lora_config = lora_config
dense = False
num_epochs = 45
config.learning_rate = 1e-4



GoogleCloudFileSystem.storage_client = gcp_client

# get the current working directory
root_dir = os.getcwd()

# define the relative path to the dataset JSON file
json_rel_path = '../' + DATASET_JSON_PATH
# json_rel_path = DATASET_JSON_PATH

# combine the root directory with the relative path
json_abs_path = os.path.join(root_dir, json_rel_path)

dataset_json = json.load(open(json_abs_path, 'r'))

all_scenes = [observation_to_scene(config, observation) for i, observation in enumerate(observation_factory(dataset_json))]
cluster_ids = [observation.cluster_id for i, observation in enumerate(observation_factory(dataset_json))]

# val_cluster_id = 9
val_cluster_id = np.unique(cluster_ids).max()
print("Using \"spatially-random\" validation cluster id: ", val_cluster_id)

# Uncomment if want new val set
# val_cluster_id = np.unique(cluster_ids).max() + 1
# for cid in np.unique(cluster_ids):
#     scene_idx = [i for i in range(len(cluster_ids)) if cluster_ids[i] == cid]
#     val_idx = random.sample(scene_idx, 1)[0]
#     cluster_ids[val_idx] = val_cluster_id
# print("Created new \"spatially-random\" validation cluster id: ", val_cluster_id)
    
if not dense:
    training_datasets = [scene_to_training_ds(config, scene) for scene, cid in zip(all_scenes, cluster_ids) if cid != val_cluster_id]
else:
    print("Running a dense training sample regime!")
    # Use a dense sliding window sampling strategy for training
    training_datasets = [scene_to_inference_ds(config, scene, full_image=False, stride=int(config.tile_size/2)) for scene, cid in zip(all_scenes, cluster_ids) if cid != val_cluster_id]
#Random sampling:
#Use 50% overlap sliding window:
# validation_datasets = [scene_to_inference_ds(config, scene, full_image=False, stride=int(config.tile_size/2)) for scene, cid in zip(all_scenes, cluster_ids) if cid == val_cluster_id]
#Random sampling:
validation_datasets = [scene_to_training_ds(config, scene) for scene, cid in zip(all_scenes, cluster_ids) if cid == val_cluster_id]


train_dataset_merged = ConcatDataset(training_datasets)
val_dataset_merged = ConcatDataset(validation_datasets)

print('Validation split cluster_id:', val_cluster_id)
print ('Training dataset size: {:4d} images'.format(len(train_dataset_merged)))
print ('Testing dataset size: {:4d}  images'.format(len(val_dataset_merged)))
perc_val = len(val_dataset_merged) / (len(val_dataset_merged) + len(train_dataset_merged))
print(f'Ratio of train:val: {1-perc_val: .2f}:{perc_val:.2f}')

#update mine class weight
# config.mine_class_loss_weight = 15
print('Mine class weight:', config.mine_class_loss_weight)


_, _, n_channels = training_datasets[0].scene.raster_source.shape
model = model_factory(
    config,
    n_channels=n_channels,
    config_lora=lora_config
)

optimizer = optimizer_factory(config, model)

name = f"SatMAE-seed{seed}-Method"

current_date = datetime.now().date().strftime("%Y-%m-%d")

if methodA:
    name += "A"
else:
    name += "B"

if dense:
    name += "-dense"

name += f"-{current_date}"

learner = learner_factory(
    config=config,
    model=model,
    optimizer=optimizer,
    train_ds=train_dataset_merged,  # for development and debugging, use training_datasets[0] or similar to speed up
    valid_ds=val_dataset_merged,  # for development and debugging, use training_datasets[1] or similar to speed up
    output_dir = f"/data/sand_mining/checkpoints/finetuned/{name}" #=expanduser(f"~/sandmining-watch/out/{name}")
)
print_trainable_parameters(learner.model)


learner.initialize_wandb_run(project="DS-v03", run_name=name)
learner.train(epochs=num_epochs)
