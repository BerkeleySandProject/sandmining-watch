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
from experiment_configs.configs import lora_config, satmae_large_config_lora_methodA
from torch.utils.data import ConcatDataset
from utils.data_management import observation_factory

gcp_client = storage.Client(project=GCP_PROJECT_NAME)


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" #to prevent cuda out of memory error
torch.cuda.empty_cache()

#For reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



config = satmae_large_config_lora_methodA
lora_config = lora_config


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
    
training_datasets = [scene_to_training_ds(config, scene) for scene, cid in zip(all_scenes, cluster_ids) if cid != val_cluster_id]
validation_datasets = [scene_to_inference_ds(config, scene, full_image=False, stride=int(config.tile_size/2)) for scene, cid in zip(all_scenes, cluster_ids) if cid == val_cluster_id]


train_dataset_merged = ConcatDataset(training_datasets)
val_dataset_merged = ConcatDataset(validation_datasets)

print('Validation split cluster_id:', val_cluster_id)
print ('Training dataset size: {:4d} images'.format(len(train_dataset_merged)))
print ('Testing dataset size: {:4d}  images'.format(len(val_dataset_merged)))

#update mine class weight
config.mine_class_loss_weight = 10
print('Updating mine class weight:', config.mine_class_loss_weight)



_, _, n_channels = training_datasets[0].scene.raster_source.shape
model = model_factory(
    config,
    n_channels=n_channels,
    config_lora=lora_config
)

optimizer = optimizer_factory(config, model)

learner = learner_factory(
    config=config,
    model=model,
    optimizer=optimizer,
    train_ds=train_dataset_merged,  # for development and debugging, use training_datasets[0] or similar to speed up
    valid_ds=val_dataset_merged,  # for development and debugging, use training_datasets[1] or similar to speed up
    output_dir=expanduser("~/sandmining-watch/out/MethodA-Seed0")
)
print_trainable_parameters(learner.model)

learner.initialize_wandb_run(project="DS-v03", run_name="SatMAE-A-SR-Seed42")
learner.train(epochs=30)
