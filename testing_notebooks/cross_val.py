import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from os.path import expanduser

from dotenv import load_dotenv
load_dotenv()

from google.cloud import storage
from project_config import GCP_PROJECT_NAME, DATASET_JSON_PATH

gcp_client = storage.Client(project=GCP_PROJECT_NAME)



import os, torch
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32" #to prevent cuda out of memory error
torch.cuda.empty_cache()



from experiment_configs.configs import lora_config, satmae_large_config_lora_methodA

config = satmae_large_config_lora_methodA
lora_config = lora_config




from torch.utils.data import ConcatDataset
import json
from utils.rastervision_pipeline import observation_to_scene, scene_to_training_ds, scene_to_validation_ds, scene_to_inference_ds
from utils.data_management import observation_factory, characterize_dataset
import random

#set the seed
random.seed(13)

# get the current working directory
root_dir = os.getcwd()

# define the relative path to the dataset JSON file
json_rel_path = '../' + DATASET_JSON_PATH

# combine the root directory with the relative path
json_abs_path = os.path.join(root_dir, json_rel_path)

dataset_json = json.load(open(json_abs_path, 'r'))

all_scenes = [observation_to_scene(config, observation) for observation in observation_factory(dataset_json)]
cluster_ids = [observation.cluster_id for observation in observation_factory(dataset_json)]

from ml.cross_validation import CrossValidator

cv = CrossValidator(all_scenes, cluster_ids, size_validation_group=1)

# cv.split_groups = cv.split_groups[2:]
# cv.splits = cv.splits[2:]


cv.cross_val(config,
             10, 
             lora_config=lora_config, 
             wandb_group_name="SatMAE Large LoRA Method A", 
             model_weights_output_folder="~/sandmining-watch/out/SatMAE-L-LoRA-MethodA-CrossVal/",
             )
